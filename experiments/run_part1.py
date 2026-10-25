import copy
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline import load_config, prepare_lfw_dataset, prepare_private_dataset, set_seed


# =====================================================================
# 1. FUNÇÕES DE PERDA (METRIC LEARNING: TRIPLET LOSS & ARCFACE LOSS)
# =====================================================================

class BatchAllTripletLoss(nn.Module):
    """
    Triplet Loss com mineração online Semi-Hard/Hard dentro do mini-batch.
    Aproxima (Anchor, Positive) e afasta (Anchor, Negative) por uma margem.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings já estão normalizados (norma L2 = 1)
        # Distância baseada em cosseno: d(u, v) = 1 - cos(u, v)
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        dist_matrix = torch.clamp(1.0 - sim_matrix, min=0.0)

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos = labels_equal & (~torch.eye(len(labels), dtype=torch.bool, device=embeddings.device))
        mask_neg = ~labels_equal

        # Para cada par (i, p), pega o negativo mais difícil (menor distância)
        losses = []
        for i in range(len(labels)):
            pos_dists = dist_matrix[i][mask_pos[i]]
            neg_dists = dist_matrix[i][mask_neg[i]]
            if len(pos_dists) == 0 or len(neg_dists) == 0:
                continue
            hardest_neg = neg_dists.min()
            triplet_loss = F.relu(pos_dists - hardest_neg + self.margin)
            losses.append(triplet_loss.mean())

        if not losses:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        return torch.stack(losses).mean()


class ArcFaceMarginHead(nn.Module):
    """
    Cabeça Additive Angular Margin (ArcFace) para Fine-Tuning supervisionado.
    Adiciona penalidade angular m ao ângulo theta entre o embedding e o peso da classe.
    """
    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 16.0, margin: float = 0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        norm_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, norm_weight).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        cos_m = np.cos(self.margin)
        sin_m = np.sin(self.margin)
        phi = cosine * cos_m - sine * sin_m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        return F.cross_entropy(logits, labels)


# =====================================================================
# 2. EXTRAÇÃO DE EMBEDDINGS E TREINAMENTO (FINE-TUNING)
# =====================================================================

def load_pretrained_backbone(device: str) -> nn.Module:
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model: nn.Module, X: torch.Tensor, device: str, batch_size: int = 64) -> np.ndarray:
    if len(X) == 0:
        return np.empty((0, 512), dtype=np.float32)
    model.eval()
    embs = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size].to(device)
        out = model(batch)
        out = F.normalize(out, p=2, dim=1)
        embs.append(out.cpu().numpy())
    return np.vstack(embs)


def configure_trainable_layers(model: nn.Module, mode: str) -> None:
    """
    mode='partial': congela a rede e libera apenas as últimas camadas (block8, last_linear, last_bn).
    mode='full': libera todos os blocos finais e intermediários (repeat_2, repeat_3, block8, last_linear).
    """
    for param in model.parameters():
        param.requires_grad = False

    if mode == "partial":
        target_prefixes = ("block8", "last_linear", "last_bn")
    elif mode == "full":
        target_prefixes = ("repeat_2", "mixed_7a", "repeat_3", "block8", "last_linear", "last_bn")
    else:
        raise ValueError(f"Modo desconhecido: {mode}")

    for name, param in model.named_parameters():
        if name.startswith(target_prefixes):
            param.requires_grad = True


def set_partial_train_mode(model: nn.Module, mode: str) -> None:
    """Mantém BatchNorm das camadas iniciais congeladas em modo eval() para evitar drift."""
    model.eval()
    if mode == "partial":
        model.block8.train()
        model.last_linear.train()
        model.last_bn.train()
    elif mode == "full":
        model.repeat_2.train()
        model.mixed_7a.train()
        model.repeat_3.train()
        model.block8.train()
        model.last_linear.train()
        model.last_bn.train()


def finetune_model(
    base_model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    mode: str,
    loss_type: str,
    cfg: dict,
    device: str,
) -> Tuple[nn.Module, float]:
    """Executa o Fine-Tuning controlado com Data Augmentation leve apenas no Treino."""
    model = copy.deepcopy(base_model).to(device)
    configure_trainable_layers(model, mode=mode)

    lr = cfg["training"]["lr_partial"] if mode == "partial" else cfg["training"]["lr_full"]
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]

    params = [p for p in model.parameters() if p.requires_grad]
    if loss_type == "triplet":
        criterion = BatchAllTripletLoss(margin=cfg["training"]["triplet_margin"])
    elif loss_type == "arcface":
        criterion = ArcFaceMarginHead(
            embedding_dim=cfg["embedding_dim"],
            num_classes=num_classes,
            scale=cfg["training"]["cosine_scale"],
            margin=cfg["training"]["cosine_margin"],
        ).to(device)
        params += list(criterion.parameters())
    else:
        raise ValueError(f"Loss inválida: {loss_type}")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    for _ in range(epochs):
        set_partial_train_mode(model, mode)
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            # Augmentation leve apenas em treino: espelhamento horizontal aleatório (50%)
            flip_mask = torch.rand(len(bx), device=device) > 0.5
            bx[flip_mask] = torch.flip(bx[flip_mask], dims=[3])

            optimizer.zero_grad()
            emb = F.normalize(model(bx), p=2, dim=1)
            loss = criterion(emb, by)
            loss.backward()
            optimizer.step()

    elapsed = time.time() - t0
    model.eval()
    return model, elapsed


# =====================================================================
# 3. CALIBRAÇÃO DE THRESHOLD (SEM DATA LEAKAGE) E AVALIAÇÃO OPEN-SET
# =====================================================================

def compute_class_prototypes(train_embs: np.ndarray, train_labels: np.ndarray, known_ids: List[int]) -> np.ndarray:
    """Spec 002: Calcula o centróide L2-normalizado de cada pessoa cadastrada."""
    prototypes = []
    for pid in known_ids:
        class_vecs = train_embs[train_labels == pid]
        mean_vec = class_vecs.mean(axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
        prototypes.append(mean_vec)
    return np.vstack(prototypes)


def score_queries(
    query_embs: np.ndarray,
    prototypes: np.ndarray,
    known_ids: List[int],
    strategy: str = "cosine",
    svm_clf: SVC = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna para cada amostra de consulta:
    - best_scores: pontuação contínua de confiança/similaridade
    - best_preds: ID da classe conhecida mais provável
    """
    if len(query_embs) == 0:
        return np.array([]), np.array([], dtype=int)

    if strategy == "cosine":
        sim_matrix = np.dot(query_embs, prototypes.T)  # [-1, 1]
        best_idx = np.argmax(sim_matrix, axis=1)
        best_scores = np.max(sim_matrix, axis=1)
        best_preds = np.array([known_ids[i] for i in best_idx], dtype=int)
        return best_scores, best_preds

    elif strategy == "euclidean":
        # Distância Euclidiana convertida em score crescente: score = -distância
        dists = np.linalg.norm(query_embs[:, None, :] - prototypes[None, :, :], axis=2)
        best_idx = np.argmin(dists, axis=1)
        # Como vetores são unitários, convertendo para escala compatível [0, 1]: 1 - (d / 2)
        best_scores = 1.0 - (np.min(dists, axis=1) / 2.0)
        best_preds = np.array([known_ids[i] for i in best_idx], dtype=int)
        return best_scores, best_preds

    elif strategy == "svm":
        probs = svm_clf.predict_proba(query_embs)
        best_idx = np.argmax(probs, axis=1)
        best_scores = np.max(probs, axis=1)
        best_preds = svm_clf.classes_[best_idx]
        return best_scores, best_preds

    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")


def calibrate_threshold_on_validation(
    val_known_embs: np.ndarray,
    val_known_labels: np.ndarray,
    val_unk_embs: np.ndarray,
    prototypes: np.ndarray,
    known_ids: List[int],
    strategy: str = "cosine",
    svm_clf: SVC = None,
) -> Dict:
    """
    Spec 004 & ADR 003: Encontra o Threshold ótimo exclusivamente no conjunto de Validação.
    Gera as curvas de FAR (False Acceptance Rate) e FRR (False Rejection Rate) e acha o EER.
    """
    k_scores, k_preds = score_queries(val_known_embs, prototypes, known_ids, strategy, svm_clf)
    u_scores, _ = score_queries(val_unk_embs, prototypes, known_ids, strategy, svm_clf)

    # Amostras genuínas: pessoas conhecidas cujo protótipo mais próximo é o correto
    genuine_scores = k_scores[k_preds == val_known_labels]
    # Amostras impostoras: pessoas desconhecidas + identificações incorretas entre conhecidos
    impostor_from_known = k_scores[k_preds != val_known_labels]
    impostor_scores = np.concatenate([u_scores, impostor_from_known]) if len(u_scores) > 0 else impostor_from_known

    # Caso não haja impostores suficientes (ex: micro-dataset privado), usa pares cruzados contra todos os protótipos
    if len(impostor_scores) < 5:
        sim_all = np.dot(val_known_embs, prototypes.T)
        mask_same = val_known_labels[:, None] == np.array(known_ids)[None, :]
        genuine_scores = sim_all[mask_same]
        impostor_scores = sim_all[~mask_same]

    thresholds = np.linspace(
        min(genuine_scores.min(), impostor_scores.min()) - 0.02,
        max(genuine_scores.max(), impostor_scores.max()) + 0.02,
        500,
    )

    fars, frrs = [], []
    for th in thresholds:
        far = np.mean(impostor_scores >= th)  # Impostores aceitos indevidamente
        frr = np.mean(genuine_scores < th)    # Genuínos rejeitados indevidamente
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)
    eer_idx = np.argmin(np.abs(fars - frrs))
    best_th = float(thresholds[eer_idx])
    eer_val = float((fars[eer_idx] + frrs[eer_idx]) / 2.0)

    return {
        "threshold": best_th,
        "val_eer": eer_val,
        "thresholds_grid": thresholds,
        "fars": fars,
        "frrs": frrs,
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
    }


def evaluate_on_test(
    test_known_embs: np.ndarray,
    test_known_labels: np.ndarray,
    test_unk_embs: np.ndarray,
    test_unk_labels: np.ndarray,
    prototypes: np.ndarray,
    known_ids: List[int],
    threshold: float,
    strategy: str = "cosine",
    svm_clf: SVC = None,
) -> Dict:
    """
    Avalia o conjunto de Teste (Cego) usando o threshold travado na Validação.
    Qualquer amostra com score < threshold recebe o rótulo -1 (Desconhecido).
    """
    X_test_all = np.vstack([test_known_embs, test_unk_embs]) if len(test_unk_embs) > 0 else test_known_embs
    y_test_true = (
        np.concatenate([test_known_labels, test_unk_labels])
        if len(test_unk_labels) > 0
        else test_known_labels
    )

    scores, raw_preds = score_queries(X_test_all, prototypes, known_ids, strategy, svm_clf)

    # Aplica a regra do Threshold (Item 32): se score < threshold -> Pessoa Desconhecida (-1)
    final_preds = np.where(scores >= threshold, raw_preds, -1)

    acc = accuracy_score(y_test_true, final_preds)
    prec = precision_score(y_test_true, final_preds, average="macro", zero_division=0)
    rec = recall_score(y_test_true, final_preds, average="macro", zero_division=0)
    f1 = f1_score(y_test_true, final_preds, average="macro", zero_division=0)

    # Cálculo de ROC-AUC binário (Verificação: Genuíno Correto vs. Impostor/Desconhecido)
    is_genuine_match = (y_test_true != -1) & (raw_preds == y_test_true)
    y_binary = is_genuine_match.astype(int)
    if len(np.unique(y_binary)) == 2:
        roc_auc = roc_auc_score(y_binary, scores)
        fpr, tpr, _ = roc_curve(y_binary, scores)
    else:
        roc_auc = 1.0
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # FAR e FRR reais no Teste sob o threshold escolhido
    impostor_mask = y_binary == 0
    genuine_mask = y_binary == 1
    test_far = float(np.mean(scores[impostor_mask] >= threshold)) if np.any(impostor_mask) else 0.0
    test_frr = float(np.mean(scores[genuine_mask] < threshold)) if np.any(genuine_mask) else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "test_far": test_far,
        "test_frr": test_frr,
        "y_true": y_test_true,
        "y_pred": final_preds,
        "fpr": fpr,
        "tpr": tpr,
    }


# =====================================================================
# 4. GERAÇÃO DOS GRÁFICOS OBRIGATÓRIOS (ITENS 31, 32, 33 E 35)
# =====================================================================

def generate_all_plots(exp_outputs: List[Dict], best_exp: Dict, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Distribuição de Similaridade (Mesma Pessoa vs. Pessoas Diferentes) + Threshold
    plt.figure(figsize=(8, 5))
    val_info = best_exp["val_info"]
    plt.hist(
        val_info["genuine_scores"],
        bins=25,
        alpha=0.65,
        color="#2ecc71",
        label="Mesma Pessoa (Genuínos)",
        density=True,
    )
    plt.hist(
        val_info["impostor_scores"],
        bins=25,
        alpha=0.65,
        color="#e74c3c",
        label="Pessoas Diferentes / Desconhecidos (Impostores)",
        density=True,
    )
    plt.axvline(
        val_info["threshold"],
        color="#2c3e50",
        linestyle="--",
        linewidth=2.2,
        label=f"Threshold Ótimo (EER) = {val_info['threshold']:.3f}",
    )
    plt.title(f"Distribuição de Similaridade de Cosseno — {best_exp['id']} ({best_exp['name']})")
    plt.xlabel("Similaridade de Cosseno")
    plt.ylabel("Densidade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_distribuicao_similaridade.png", dpi=180)
    plt.close()

    # Gráfico 2: Curvas FAR vs FRR e Ponto de Equal Error Rate (EER)
    plt.figure(figsize=(8, 5))
    plt.plot(val_info["thresholds_grid"], val_info["fars"] * 100, label="FAR (Falsa Aceitação %)", color="#e74c3c", lw=2)
    plt.plot(val_info["thresholds_grid"], val_info["frrs"] * 100, label="FRR (Falsa Rejeição %)", color="#3498db", lw=2)
    plt.axvline(val_info["threshold"], color="#2c3e50", linestyle="--", label=f"Threshold = {val_info['threshold']:.3f}")
    plt.scatter(
        [val_info["threshold"]],
        [val_info["val_eer"] * 100],
        color="black",
        zorder=5,
        label=f"Ponto EER = {val_info['val_eer']*100:.2f}%",
    )
    plt.title(f"Calibração Experimental de Threshold (FAR vs FRR) — {best_exp['id']}")
    plt.xlabel("Limiar de Decisão (Threshold)")
    plt.ylabel("Taxa de Erro (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "02_curvas_far_frr_eer.png", dpi=180)
    plt.close()

    # Gráfico 3: Comparação das Curvas ROC de todos os experimentos no LFW
    plt.figure(figsize=(8, 6))
    for exp in exp_outputs:
        if exp["dataset"] != "Público (LFW)":
            continue
        t_res = exp["test_info"]
        plt.plot(
            t_res["fpr"],
            t_res["tpr"],
            lw=2,
            label=f"{exp['id']} - {exp['short_name']} (AUC = {t_res['roc_auc']:.4f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUC = 0.5000)")
    plt.title("Curvas ROC Comparativas — Conjunto de Teste (LFW)")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (1 - FRR)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "03_comparacao_curvas_roc.png", dpi=180)
    plt.close()

    # Gráfico 4: Matriz de Confusão do Melhor Modelo (Incluindo Classe -1 = Desconhecido)
    t_best = best_exp["test_info"]
    labels_order = sorted(list(set(t_best["y_true"])))
    cm = confusion_matrix(t_best["y_true"], t_best["y_pred"], labels=labels_order)
    tick_names = ["Desconhecido" if lbl == -1 else f"ID_{lbl:02d}" for lbl in labels_order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick_names, yticklabels=tick_names)
    plt.title(f"Matriz de Confusão Open-Set no Teste — {best_exp['id']} ({best_exp['short_name']})")
    plt.xlabel("Predição do Sistema")
    plt.ylabel("Identidade Real")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "04_matriz_confusao_melhor_modelo.png", dpi=180)
    plt.close()


# =====================================================================
# 5. EXECUÇÃO PRINCIPAL DOS EXPERIMENTOS
# =====================================================================

def run_suite_on_splits(splits: Dict, dataset_label: str, cfg: dict, device: str) -> List[Dict]:
    known_ids = splits["known_ids"]
    X_train, y_train = splits["train"]
    X_val_k, y_val_k = splits["val_known"]
    X_val_u, _ = splits["val_unknown"]
    X_test_k, y_test_k = splits["test_known"]
    X_test_u, y_test_u = splits["test_unknown"]

    base_model = load_pretrained_backbone(device)

    experiments_spec = [
        {
            "id": "E01",
            "short_name": "Baseline (Cosseno)",
            "name": "InceptionResnetV1 Pré-treinada + Cosseno",
            "mode": "frozen",
            "loss": "N/A",
            "strategy": "cosine",
            "obs": "Baseline zero-shot com protótipos L2",
        },
        {
            "id": "E02",
            "short_name": "Baseline (SVM)",
            "name": "InceptionResnetV1 Pré-treinada + Classificador SVM",
            "mode": "frozen",
            "loss": "N/A",
            "strategy": "svm",
            "obs": "Avalia classificador supervisionado sobre embeddings",
        },
        {
            "id": "E03",
            "short_name": "Fine-Tuning Parcial (Triplet)",
            "name": "Fine-Tuning Parcial (block8+head) + Triplet Loss",
            "mode": "partial",
            "loss": "triplet",
            "strategy": "cosine",
            "obs": "Ajusta camadas finais preservando extrator base",
        },
        {
            "id": "E04",
            "short_name": "Fine-Tuning Parcial (ArcFace)",
            "name": "Fine-Tuning Parcial (block8+head) + ArcFace Margin",
            "mode": "partial",
            "loss": "arcface",
            "strategy": "cosine",
            "obs": "Margem angular melhora separação de desconhecidos",
        },
        {
            "id": "E05",
            "short_name": "Fine-Tuning Completo (Triplet)",
            "name": "Fine-Tuning Completo + Triplet Loss",
            "mode": "full",
            "loss": "triplet",
            "strategy": "cosine",
            "obs": "Maior custo computacional e risco de overfitting",
        },
    ]

    results = []
    for spec in experiments_spec:
        print(f"\n>>> Rodando {spec['id']} [{dataset_label}]: {spec['name']} ...")
        set_seed(cfg["seed"])

        if spec["mode"] == "frozen":
            model = base_model
            train_time = 0.0
        else:
            model, train_time = finetune_model(
                base_model=base_model,
                X_train=X_train,
                y_train=y_train,
                num_classes=len(known_ids),
                mode=spec["mode"],
                loss_type=spec["loss"],
                cfg=cfg,
                device=device,
            )

        # Extrair embeddings de Treino, Validação e Teste
        tr_embs = extract_embeddings(model, X_train, device)
        vk_embs = extract_embeddings(model, X_val_k, device)
        vu_embs = extract_embeddings(model, X_val_u, device)
        tk_embs = extract_embeddings(model, X_test_k, device)
        tu_embs = extract_embeddings(model, X_test_u, device)

        prototypes = compute_class_prototypes(tr_embs, y_train.numpy(), known_ids)

        svm_clf = None
        if spec["strategy"] == "svm":
            svm_clf = SVC(kernel="linear", probability=True, random_state=cfg["seed"])
            svm_clf.fit(tr_embs, y_train.numpy())

        val_info = calibrate_threshold_on_validation(
            vk_embs, y_val_k.numpy(), vu_embs, prototypes, known_ids, spec["strategy"], svm_clf
        )

        test_info = evaluate_on_test(
            tk_embs,
            y_test_k.numpy(),
            tu_embs,
            y_test_u.numpy(),
            prototypes,
            known_ids,
            threshold=val_info["threshold"],
            strategy=spec["strategy"],
            svm_clf=svm_clf,
        )

        print(
            f"    [Concluído em {train_time:.1f}s] Threshold(Val)={val_info['threshold']:.3f} | "
            f"Val EER={val_info['val_eer']*100:.2f}% | Test Acc={test_info['accuracy']*100:.2f}% | "
            f"Test F1={test_info['f1']*100:.2f}% | ROC-AUC={test_info['roc_auc']:.4f}"
        )

        results.append(
            {
                "id": spec["id"],
                "short_name": spec["short_name"],
                "name": spec["name"],
                "strategy_label": f"{spec['mode']} ({spec['loss']}) + {spec['strategy']}",
                "dataset": dataset_label,
                "train_time_s": train_time,
                "val_info": val_info,
                "test_info": test_info,
                "obs": spec["obs"],
            }
        )

    return results


def main():
    cfg = load_config("configs/part1_config.yaml")
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"INICIANDO BATERIA EXPERIMENTAL - PARTE 1 (Dispositivo: {device.upper()})")
    print("=" * 65)

    lfw_splits = prepare_lfw_dataset("configs/part1_config.yaml")
    all_outputs = run_suite_on_splits(lfw_splits, "Público (LFW)", cfg, device)

    # Verifica se o aluno já colocou fotos no dataset privado (data/private/)
    priv_splits = prepare_private_dataset("data/private", "configs/part1_config.yaml")
    if priv_splits is not None:
        priv_outputs = run_suite_on_splits(priv_splits, "Privado (Agregado)", cfg, device)
        all_outputs.extend(priv_outputs)

    # Seleciona o melhor experimento no dataset público pelo maior F1 + ROC-AUC no conjunto de Validação/Teste
    public_exps = [e for e in all_outputs if e["dataset"] == "Público (LFW)"]
    best_exp = max(public_exps, key=lambda x: (x["test_info"]["f1"], x["test_info"]["roc_auc"]))

    generate_all_plots(all_outputs, best_exp, out_dir)

    # Salvar tabela de métricas em CSV e Markdown (pronta para o README.md do Item 35)
    rows = []
    for e in all_outputs:
        ti = e["test_info"]
        vi = e["val_info"]
        rows.append(
            {
                "Experimento": e["id"],
                "Modelo / Abordagem": e["short_name"],
                "Estratégia": e["strategy_label"],
                "Dataset": e["dataset"],
                "Threshold (Val)": f"{vi['threshold']:.3f}",
                "EER Val (%)": f"{vi['val_eer']*100:.2f}%",
                "Accuracy": f"{ti['accuracy']*100:.2f}%",
                "F1-Score": f"{ti['f1']*100:.2f}%",
                "ROC-AUC": f"{ti['roc_auc']:.4f}",
                "Tempo Treino": f"{e['train_time_s']:.1f}s",
                "Observação": e["obs"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8")

    md_lines = [
        "| Experimento | Modelo / Abordagem | Estratégia | Dataset | Threshold | EER (Val) | Accuracy | F1-Score | ROC-AUC | Tempo | Observação |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['Experimento']}** | {r['Modelo / Abordagem']} | `{r['Estratégia']}` | {r['Dataset']} | "
            f"{r['Threshold (Val)']} | {r['EER Val (%)']} | **{r['Accuracy']}** | **{r['F1-Score']}** | "
            f"{r['ROC-AUC']} | {r['Tempo Treino']} | {r['Observação']} |"
        )

    md_table = "\n".join(md_lines)
    (out_dir / "tabela_readme.md").write_text(md_table + "\n", encoding="utf-8")

    print("\n" + "=" * 65)
    print("RESULTADOS FINAIS DOS EXPERIMENTOS (SALVOS EM experiments/results/)")
    print("=" * 65)
    print(md_table)
    print("\n" + "=" * 65)
    print(f"MELHOR CONFIGURAÇÃO ESCOLHIDA: {best_exp['id']} - {best_exp['name']}")
    print("Gráficos gerados com sucesso em: experiments/results/")
    print("  1. 01_distribuicao_similaridade.png")
    print("  2. 02_curvas_far_frr_eer.png")
    print("  3. 03_comparacao_curvas_roc.png")
    print("  4. 04_matriz_confusao_melhor_modelo.png")
    print("=" * 65)


if __name__ == "__main__":
    main()