# ADR 002: Isolamento e Não-Publicação de Dados Biométricos

## Decisão
Bloquear via `.gitignore` qualquer imagem (`.jpg`, `.png`), embedding (`.npy`, `.pt`) e checkpoint (`.pth`) derivado de dados locais, versionando apenas código e métricas estatísticas agregadas.
