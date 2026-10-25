import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    """Fixa todas as sementes aleatórias para garantir reprodutibilidade (Item 34)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path: str = "configs/part1_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class FaceDetectorAligner:
    """Atende à Spec 001: Detecta e alinha faces em 160x160 usando MTCNN."""

    def __init__(self, image_size: int = 160, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=14,
            keep_all=False,
            select_largest=True,
            post_process=True,
            device=self.device,
        )

    def detect_and_align(self, img: Image.Image) -> torch.Tensor:
        """Detecta a face com MTCNN; se a foto já for um recorte fechado onde o detector falhe, aplica fallback."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        with torch.no_grad():
            face_tensor = self.mtcnn(img)
        if face_tensor is not None:
            return face_tensor.cpu()

        # Fallback seguro para recortes já centralizados
        img_resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - 127.5) / 128.0


def build_openset_splits(
    person_faces: Dict[int, List[torch.Tensor]],
    unknown_ratio: float = 0.20,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Dict:
    """
    Separa os dados sem Data Leakage (Itens 29, 30 e 32):
    - Identidades Conhecidas (80%): divididas em Treino (60%), Validação (20%) e Teste (20%).
    - Identidades Desconhecidas/Impostores (20%, rótulo -1): divididas apenas entre Validação e Teste.
    """
    set_seed(seed)
    all_ids = sorted(list(person_faces.keys()))
    num_unknown = int(round(len(all_ids) * unknown_ratio))

    if num_unknown > 0 and len(all_ids) - num_unknown >= 2:
        known_ids = all_ids[:-num_unknown]
        unknown_ids = all_ids[-num_unknown:]
    else:
        known_ids = all_ids
        unknown_ids = []

    X_train, y_train = [], []
    X_val_known, y_val_known = [], []
    X_test_known, y_test_known = [], []

    for pid in known_ids:
        faces = person_faces[pid]
        n = len(faces)
        idx = list(range(n))
        random.shuffle(idx)

        n_train = max(2, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        for i in idx[:n_train]:
            X_train.append(faces[i])
            y_train.append(pid)
        for i in idx[n_train : n_train + n_val]:
            X_val_known.append(faces[i])
            y_val_known.append(pid)
        for i in idx[n_train + n_val :]:
            X_test_known.append(faces[i])
            y_test_known.append(pid)

    X_val_unk, y_val_unk = [], []
    X_test_unk, y_test_unk = [], []

    for uid in unknown_ids:
        faces = person_faces[uid]
        mid = max(1, len(faces) // 2)
        for f in faces[:mid]:
            X_val_unk.append(f)
            y_val_unk.append(-1)
        for f in faces[mid:]:
            X_test_unk.append(f)
            y_test_unk.append(-1)

    return {
        "known_ids": known_ids,
        "unknown_ids": unknown_ids,
        "train": (torch.stack(X_train), torch.tensor(y_train, dtype=torch.long)),
        "val_known": (torch.stack(X_val_known), torch.tensor(y_val_known, dtype=torch.long)),
        "test_known": (torch.stack(X_test_known), torch.tensor(y_test_known, dtype=torch.long)),
        "val_unknown": (
            torch.stack(X_val_unk) if X_val_unk else torch.empty((0, 3, 160, 160)),
            torch.tensor(y_val_unk, dtype=torch.long),
        ),
        "test_unknown": (
            torch.stack(X_test_unk) if X_test_unk else torch.empty((0, 3, 160, 160)),
            torch.tensor(y_test_unk, dtype=torch.long),
        ),
    }


def prepare_lfw_dataset(config_path: str = "configs/part1_config.yaml") -> Dict:
    """Baixa o dataset público LFW, alinha com MTCNN e salva os splits em cache local."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    cache_file = Path("data/public/lfw_cropped_cache.pt")
    if cache_file.exists():
        print("[INFO] Carregando faces alinhadas do LFW a partir do cache local...")
        return torch.load(cache_file, weights_only=False)

    print("[INFO] Baixando dataset público LFW via scikit-learn...")
    lfw = fetch_lfw_people(
        data_home="data/public",
        min_faces_per_person=cfg["lfw_dataset"]["min_faces_per_person"],
        color=True,
        resize=1.0,
    )

    unique_ids, counts = np.unique(lfw.target, return_counts=True)
    top_indices = np.argsort(-counts)[: cfg["lfw_dataset"]["max_identities"]]
    selected_targets = set(unique_ids[top_indices])

    detector = FaceDetectorAligner(image_size=cfg["image_size"])
    person_faces: Dict[int, List[torch.Tensor]] = {int(t): [] for t in selected_targets}

    print("[INFO] Alinhando faces do LFW com MTCNN...")
    for img_array, target in tqdm(zip(lfw.images, lfw.target), total=len(lfw.target)):
        t_id = int(target)
        if t_id not in selected_targets or len(person_faces[t_id]) >= 25:
            continue
        img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.0 else img_array.astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        person_faces[t_id].append(detector.detect_and_align(pil_img))

    valid_persons = sorted([p for p, faces in person_faces.items() if len(faces) >= 10])
    remapped_faces = {new_id: person_faces[old_id] for new_id, old_id in enumerate(valid_persons)}

    splits = build_openset_splits(
        remapped_faces,
        unknown_ratio=cfg["lfw_dataset"]["unknown_identities_ratio"],
        train_ratio=cfg["splits"]["train_ratio"],
        val_ratio=cfg["splits"]["val_ratio"],
        seed=cfg["seed"],
    )

    torch.save(splits, cache_file)
    print(f"[OK] Cache salvo em {cache_file}")
    return splits


def prepare_private_dataset(
    private_dir: str = "data/private",
    config_path: str = "configs/part1_config.yaml",
) -> Optional[Dict]:
    """Lê a pasta data/private/ (se houver fotos), aplica MTCNN e cria os splits locais."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    priv_path = Path(private_dir)
    if not priv_path.exists():
        return None

    subdirs = sorted([d for d in priv_path.iterdir() if d.is_dir()])
    if len(subdirs) < 2:
        return None

    detector = FaceDetectorAligner(image_size=cfg["image_size"])
    person_faces: Dict[int, List[torch.Tensor]] = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for idx, folder in enumerate(subdirs):
        img_files = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_exts])
        faces = [detector.detect_and_align(Image.open(f)) for f in img_files]
        if len(faces) >= 5:
            person_faces[idx] = faces

    if len(person_faces) < 2:
        return None

    remapped = {new_i: faces for new_i, (_, faces) in enumerate(sorted(person_faces.items()))}
    unk_ratio = 0.25 if len(remapped) >= 4 else 0.0
    return build_openset_splits(
        remapped,
        unknown_ratio=unk_ratio,
        train_ratio=cfg["splits"]["train_ratio"],
        val_ratio=cfg["splits"]["val_ratio"],
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    splits = prepare_lfw_dataset()
    print("\n" + "=" * 55)
    print("RESUMO DO DATASET PÚBLICO (LFW) - ZERO DATA LEAKAGE")
    print("=" * 55)
    print(f"Identidades Conhecidas (Cadastradas) : {len(splits['known_ids'])}")
    print(f"Identidades Desconhecidas (Open-Set) : {len(splits['unknown_ids'])}")
    print(f"Amostras de Treino (Conhecidos)      : {len(splits['train'][0])}")
    print(f"Amostras de Validação (Conhecidos)   : {len(splits['val_known'][0])}")
    print(f"Amostras de Validação (Desconhecidos): {len(splits['val_unknown'][0])}")
    print(f"Amostras de Teste (Conhecidos)       : {len(splits['test_known'][0])}")
    print(f"Amostras de Teste (Desconhecidos)    : {len(splits['test_unknown'][0])}")
    print("=" * 55)