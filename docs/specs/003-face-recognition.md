# Spec 003: Comparação e Reconhecimento Facial

## Objetivo
Associar uma face de consulta à identidade correta ou rejeitá-la com base em métricas de similaridade.

## Requisitos e Critérios de Aceitação
- Suportar comparação por Similaridade de Cosseno, Distância Euclidiana e classificadores supervisionados (SVM/k-NN).
- Retornar a identidade cadastrada apenas se a pontuação atingir ou superar o limiar `threshold`.
