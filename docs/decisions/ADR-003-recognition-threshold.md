# ADR 003: Calibração de Threshold sem Data Leakage

## Decisão
O limiar de decisão (`threshold`) jamais é ajustado no conjunto de Teste. Ele é calculado sobre pares genuínos e impostores do conjunto de Validação no ponto de Equal Error Rate (EER).
