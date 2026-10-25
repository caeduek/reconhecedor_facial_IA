# Sistema de Reconhecimento Facial — Parte 1 (Experimentação e Fine-Tuning)

Trabalho prático da disciplina de **Inteligência Artificial** (FACOM/UFMS).  
Este repositório implementa o ciclo experimental de detecção, alinhamento, extração de *embeddings*, *fine-tuning* com *Metric Learning* e calibração de limiar (*Open-Set Threshold*) seguindo **Spec-Driven Development** e conformidade estrita com a **LGPD (Lei nº 13.709/2018)**.

---

## 1. Arquitetura do Pipeline

```text
Imagem RGB -> Detecção e Alinhamento (MTCNN 160x160) -> Backbone (InceptionResnetV1) -> Embedding L2 (512-d) -> Comparação + Threshold (EER) -> Identidade ou Desconhecido