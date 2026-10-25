# ADR 001: Arquitetura Base e Estratégias de Fine-Tuning

## Decisão
Utilizar `InceptionResnetV1` (FaceNet pré-treinada no VGGFace2) como backbone principal, avaliando:
1. Modelo congelado (Baseline);
2. Fine-tuning parcial (cabeça de projeção e últimas camadas desbloqueadas) com Triplet Loss e Cosine Margin (ArcFace-style);
3. Fine-tuning completo.
