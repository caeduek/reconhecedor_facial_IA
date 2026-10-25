# Spec 002: Cadastro e Protótipos de Identidade

## Objetivo
Representar cada pessoa cadastrada através de um vetor protótipo (centróide) normalizado na esfera unitária de 512 dimensões.

## Requisitos e Critérios de Aceitação
- Extrair os embeddings das amostras de treino de uma identidade conhecida.
- Calcular a média vetorial e aplicar normalização L2 (`norm = 1.0`).
