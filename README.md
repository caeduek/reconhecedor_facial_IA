| Experimento | Modelo / Abordagem | Estratégia | Dataset | Threshold | EER (Val) | Accuracy | F1-Score | ROC-AUC | Tempo | Observação |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E01** | Baseline (Cosseno) | `frozen (N/A) + cosine` | Público (LFW) | 0.584 | 0.00% | **99.39%** | **99.43%** | 0.9998 | 0.0s | Baseline zero-shot com protótipos L2 |
| **E02** | Baseline (SVM) | `frozen (N/A) + svm` | Público (LFW) | 0.204 | 0.00% | **98.18%** | **98.66%** | 0.9998 | 0.0s | Avalia classificador supervisionado sobre embeddings |
| **E03** | Fine-Tuning Parcial (Triplet) | `partial (triplet) + cosine` | Público (LFW) | 0.752 | 0.00% | **96.36%** | **97.62%** | 1.0000 | 55.9s | Ajusta camadas finais preservando extrator base |
| **E04** | Fine-Tuning Parcial (ArcFace) | `partial (arcface) + cosine` | Público (LFW) | 0.849 | 1.83% | **97.58%** | **97.84%** | 0.9991 | 54.9s | Margem angular melhora separação de desconhecidos |
| **E05** | Fine-Tuning Completo (Triplet) | `full (triplet) + cosine` | Público (LFW) | 0.749 | 3.67% | **99.39%** | **99.53%** | 0.9995 | 105.7s | Maior custo computacional e risco de overfitting |
| **E01** | Baseline (Cosseno) | `frozen (N/A) + cosine` | Privado (Agregado) | 0.808 | 50.00% | **40.00%** | **21.21%** | 0.5221 | 0.0s | Baseline zero-shot com protótipos L2 |
| **E02** | Baseline (SVM) | `frozen (N/A) + svm` | Privado (Agregado) | 0.495 | 38.69% | **48.00%** | **38.35%** | 0.7361 | 0.0s | Avalia classificador supervisionado sobre embeddings |
| **E03** | Fine-Tuning Parcial (Triplet) | `partial (triplet) + cosine` | Privado (Agregado) | 0.835 | 7.69% | **76.00%** | **74.11%** | 0.9467 | 8.2s | Ajusta camadas finais preservando extrator base |
| **E04** | Fine-Tuning Parcial (ArcFace) | `partial (arcface) + cosine` | Privado (Agregado) | 0.914 | 23.08% | **76.00%** | **74.79%** | 0.8896 | 7.7s | Margem angular melhora separação de desconhecidos |
| **E05** | Fine-Tuning Completo (Triplet) | `full (triplet) + cosine` | Privado (Agregado) | 0.550 | 15.48% | **80.00%** | **81.40%** | 0.9744 | 15.3s | Maior custo computacional e risco de overfitting |
