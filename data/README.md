# Guia de Dados do Projeto (Conformidade LGPD)

## 1. Dataset Público (LFW - Labeled Faces in the Wild)
- **Fonte Oficial:** http://vis-www.cs.umass.edu/lfw/ (acessado via `scikit-learn`).
- **Finalidade:** Benchmark acadêmico, treinamento/fine-tuning e calibração experimental de threshold na Parte 1.
- **Licença e Restrições:** Uso acadêmico de pesquisa. As imagens não são redistribuídas neste repositório Git; o pipeline realiza o download local automático em `data/public/` (ignorado pelo Git).

## 2. Dataset Privado do Grupo
- **Localização Local:** `data/private/` (estritamente bloqueado pelo `.gitignore`).
- **Regra de Coleta:** Fotografias obtidas exclusivamente de adultos (maiores de 18 anos) mediante assinatura prévia do Termo de Consentimento (Anexo A), guardado em local privado fora do repositório.
- **Estrutura Esperada Localmente:**
  - `data/private/subject_01/*.jpg`
  - `data/private/subject_02/*.jpg`
  - `data/private/subject_03/*.jpg`
