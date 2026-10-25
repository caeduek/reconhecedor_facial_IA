# Spec 004: Reconhecimento Open-Set (Pessoa Desconhecida)

## Objetivo
Rejeitar faces de indivíduos não cadastrados no banco de dados (impostores).

## Requisitos e Critérios de Aceitação
- Calibrar o `threshold` exclusivamente no conjunto de Validação minimizando a diferença entre FAR (False Acceptance Rate) e FRR (False Rejection Rate) — ponto de EER.
- Qualquer amostra no Teste com similaridade abaixo do `threshold` deve receber o rótulo `UNKNOWN` (`-1`).
