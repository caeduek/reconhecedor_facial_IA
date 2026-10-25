# Spec 001: Detecção e Alinhamento Facial

## Objetivo
Localizar faces humanas em imagens RGB e retornar o recorte alinhado de `160x160` pixels.

## Requisitos e Critérios de Aceitação
- Utilizar `MTCNN` para detectar bounding box e 5 landmarks faciais.
- Normalizar os pixels no intervalo `[-1, 1]` esperado pela `InceptionResnetV1`.
- Caso nenhuma face seja detectada na imagem, retornar `None` de forma segura.
