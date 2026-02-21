# Detecção de Fraudes em Transações Financeiras com XGBoost

## Descrição do Projeto
Este repositório contém a implementação de um modelo de Machine Learning focado na detecção de transações financeiras fraudulentas, utilizando o algoritmo XGBoost. O projeto aborda um problema clássico de classificação binária em dados tabulares, caracterizado por um forte desbalanceamento entre as classes de transações legítimas e fraudulentas. A solução segue um pipeline completo que abrange desde a análise exploratória dos dados (EDA), pré-processamento e engenharia de features, até o treinamento, avaliação do modelo e geração de um arquivo de submissão no formato exigido pela plataforma Kaggle.

## Contexto do Desafio
Este trabalho foi desenvolvido como parte de um desafio técnico individual proposto pela Liga Acadêmica de Inteligência Artificial (Ligia), focado em Aprendizado de Máquina. O objetivo principal é construir um modelo robusto capaz de identificar fraudes com alta precisão. A avaliação do desempenho do modelo é baseada na métrica ROC AUC (Receiver Operating Characteristic Area Under the Curve), que é particularmente relevante para problemas com classes desbalanceadas, pois avalia a capacidade de distinção do modelo em diferentes limiares de decisão.

## Estrutura do Repositório
A estrutura principal do repositório deve ser organizada para facilitar a execução e a compreensão do projeto:
- `main_fraud_detection.py` ou `XGBOOST.ipynb`: Arquivo principal com o código Python que implementa o pipeline completo (EDA, pré-processamento, treinamento, avaliação e geração de submissão).
- `requirements.txt`: Lista de todas as dependências do Python necessárias para a execução do projeto.
- `data/`: Pasta (opcional) para armazenar os datasets `train.csv` e `test.csv`.
- `submission.csv`: O arquivo de saída gerado pelo script, contendo as previsões para a plataforma Kaggle.

## Pré-requisitos
Para executar este projeto, você precisará ter instalado:
- **Python**: Versão 3.8 ou superior é recomendada.
- **pip**: Gerenciador de pacotes do Python.
- **Ambiente Virtual (opcional, mas altamente recomendado)**: Para gerenciar as dependências do projeto isoladamente.

## Configuração do Ambiente
Siga os passos abaixo para configurar o seu ambiente de desenvolvimento:

1.  **Clone o Repositório (se aplicável):**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <NOME_DO_SEU_REPOSITORIO>
    ```

2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv venv
    # No Linux/macOS:
    source venv/bin/activate
    # No Windows:
    .\venv\Scripts\activate
    ```

3.  **Instale as Dependências:**
    Instale as bibliotecas necessárias usando o `requirements.txt` fornecido ou instalando-as manualmente.
    ```bash
    pip install -r requirements.txt
    ```
    Caso o arquivo `requirements.txt` não esteja disponível, você pode instalar as dependências manualmente:
    ```bash
    pip install pandas numpy scikit-learn xgboost matplotlib seaborn
    ```

## Aquisição dos Dados
Os datasets `train.csv` e `test.csv` não estão incluídos diretamente neste repositório devido ao seu tamanho e à natureza do desafio. Eles devem ser baixados da página oficial do desafio de detecção de fraudes na plataforma Kaggle. Após o download, coloque ambos os arquivos na raiz do projeto (ou na pasta `data/` se você tiver criado uma) para que o script possa acessá-los.

## Instruções de Execução do Código
O pipeline completo para detecção de fraudes pode ser executado através do notebook Jupyter/Google Colab ou de um script Python. As etapas a seguir detalham o processo:

### 1. Carregamento dos Dados
O script inicia carregando os arquivos `train.csv` e `test.csv` utilizando a biblioteca `pandas`.
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os datasets (certifique-se de que train.csv e test.csv estão na mesma pasta)
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print("Dimensões do treino:", df_train.shape)
print("Dimensões do teste:", df_test.shape)
```

### 2. Análise Exploratória de Dados (EDA)
Esta etapa envolve a inspeção inicial dos dados para entender suas características, padrões e desafios, como desbalanceamento de classes, valores ausentes e correlações.
- Visualização das primeiras linhas (`df_train.head()`).
- Informações sobre tipos de dados e valores não nulos (`df_train.info()`).
- Estatísticas descritivas (`df_train.describe()`).
- Verificação do desbalanceamento da coluna alvo (`'Class'`) com `value_counts(normalize=True)` e visualização com `sns.countplot`.
- Análise de valores ausentes (`df_train.isnull().sum()`).
- Visualização da matriz de correlação (`sns.heatmap`).

### 3. Pré-processamento e Engenharia de Features

O código realiza o alinhamento das colunas de treino e teste para garantir consistência:
```python
# Assumindo que a coluna alvo se chama 'Class' (ajuste se for outro nome)
target_column = 'Class'

# Definir Features (X) e Alvo (y)
train_labels = df_train[target_column]
train_ids = df_train['id'] # Salvar IDs para uso futuro se necessário
test_ids = df_test['id']

# Remover colunas que não são features (neste caso, a coluna alvo e IDs)
df_train_features = df_train.drop(columns=[target_column, 'id'])
df_test_features = df_test.drop(columns=['id'])

# Garantir que ambos os dataframes tenham as mesmas colunas
common_cols = list(set(df_train_features.columns) & set(df_test_features.columns))
df_train_processed = df_train_features[common_cols]
df_test_processed = df_test_features[common_cols]

X = df_train_processed
y = train_labels
```

### 4. Modelagem com XGBoost
A modelagem é feita usando `xgb.XGBClassifier`. O parâmetro `scale_pos_weight` é crucial para lidar com o desbalanceamento das classes, dando maior peso aos erros na classe minoritária (fraudes).
```python
# Calcular o scale_pos_weight para lidar com o desbalanceamento de classes
ratio = float(np.sum(y == 0)) / np.sum(y == 1)

# Instanciar o modelo XGBoost com parâmetros iniciais robustos
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic', # Objetivo de classificação binária
    eval_metric='auc',           # Métrica de avaliação
    n_estimators=1000,           # Número de árvores
    learning_rate=0.05,          # Taxa de aprendizado
    max_depth=5,                 # Profundidade máxima das árvores
    subsample=0.8,               # Fração de amostras por árvore
    colsample_bytree=0.8,        # Fração de features por árvore
    gamma=0.1,                   # Parâmetro de regularização
    scale_pos_weight=ratio,      # CRUCIAL para dados desbalanceados
    use_label_encoder=False,     # Evitar warnings
    n_jobs=-1,                   # Usar todos os cores do CPU
    random_state=42
)

# Criar um conjunto de validação para monitorar o treinamento e evitar overfitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Treinar o modelo
print("Iniciando o treinamento do modelo...")
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False) # verbose=False para uma saída mais limpa
print("Treinamento concluído.")
```

### 5. Avaliação do Modelo
A performance do modelo é avaliada no conjunto de validação:
```python
# Fazer previsões no conjunto de validação
preds_val = xgb_clf.predict_proba(X_val)[:, 1]

# Calcular o AUC-ROC
auc_score = roc_auc_score(y_val, preds_val)
print(f"AUC no conjunto de validação: {auc_score:.5f}")

# Analisar a Matriz de Confusão
preds_binary = (preds_val > 0.5).astype(int)
print("\nMatriz de Confusão:")
print(confusion_matrix(y_val, preds_binary))
print("\nRelatório de Classificação:")
print(classification_report(y_val, preds_binary))
```

### 6. Geração do Arquivo de Submissão para o Kaggle
Finalmente, o modelo treinado é usado para prever as probabilidades de fraude no conjunto de dados de teste (`df_test`), e o arquivo `submission.csv` é gerado no formato especificado pelo Kaggle.
```python
# Fazer previsões no conjunto de teste real usando o modelo treinado
test_predictions = xgb_clf.predict_proba(df_test_processed)[:, 1]

# Criar o arquivo de submissão no formato exigido pelo Kaggle
submission = pd.DataFrame({
    'TransactionID': test_ids,
    target_column: test_predictions
})

# Salvar o arquivo de submissão
submission.to_csv('submission.csv', index=False)

print("\nArquivo 'submission.csv' gerado com sucesso!")
print(f"As primeiras 5 linhas do arquivo de submissão:\n{submission.head()}")
```

## Execução Completa (Script Principal)
Para executar o pipeline completo e gerar o arquivo de submissão, certifique-se de que os arquivos `train.csv` e `test.csv` estão na pasta correta e execute o script Python:
```bash
python main_fraud_detection.py
```
Ou, se você estiver usando um notebook Jupyter/Colab, execute todas as células do notebook `XGBOOST.ipynb`.

## Instruções para Submissão no Kaggle
Após a execução bem-sucedida do código, o arquivo `submission.csv` será gerado na raiz do seu projeto. Para submetê-lo ao desafio no Kaggle:

1.  **Acesse a Plataforma Kaggle:** Navegue até a página do desafio de detecção de fraudes.
2.  **Suba o Arquivo:** Na seção de submissão, faça o upload do arquivo `submission.csv` gerado.
3.  **Identificação:** Lembre-se de que seu *Display Name* ou *Team Name* na plataforma Kaggle deve corresponder ao seu nome real, conforme as regras do processo seletivo, para garantir a correta atribuição de sua pontuação.

## Reprodutibilidade
A reprodutibilidade é garantida pelo arquivo `requirements.txt`, que lista todas as dependências do projeto com suas versões específicas. Ao instalar as bibliotecas a partir deste arquivo, você assegura que o ambiente de execução será o mesmo do desenvolvimento. O uso de `random_state=42` no modelo XGBoost e no `train_test_split` também contribui para a consistência dos resultados.
