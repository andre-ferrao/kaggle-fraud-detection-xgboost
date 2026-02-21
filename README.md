# Detecção de Fraudes em Transações Financeiras com XGBoost

Este repositório contém o código para a solução do desafio de detecção de fraudes em transações financeiras do Kaggle, utilizando o algoritmo XGBoost.

## Estrutura do Projeto
- `src/`: Contém o código-fonte principal (`main.py`) para carregamento, pré-processamento, treinamento, avaliação e geração de submissão.
- `data/`: Pasta para os datasets `train.csv` e `test.csv` (não versionados).
- `notebooks/`: Onde notebooks de experimentação podem ser armazenados.
- `models/`: Para salvar modelos treinados.
- `results/`: Para armazenar gráficos, relatórios e o arquivo `submission.csv`.

## Como Rodar

### 1. Configuração do Ambiente
Certifique-se de ter Python 3.x instalado. Crie um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate # No Linux/macOS
venv\Scripts\activate # No Windows