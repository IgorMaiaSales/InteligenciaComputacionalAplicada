import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ========================================================================
# Lendo o data frame

df = pd.read_csv("abalone.csv")

# ========================================================================
# Selecionando as variáveis independentes X e as variáveis dependentes Y

# ':' = Seleciona todas as linhas
# ':8' = Seleciona as colunas do ínicio (coluna 0) até a coluna 8
# (não inclui a coluna 8)
X = df.iloc[:, :8].values

# ':' = Seleciona todas as linhas
# ':8' = Seleciona a coluna 8(Rings)(Foi formando como array vertical)
Y = df.iloc[:, 8].values

# ========================================================================

# Existe um processo para tratar dados faltando
# Mas não acreditor que seja necessário

# ========================================================================
# Processo de Tratamento da variáveis categoricas (Male, Female e Infant)

# Abordagem utilizada: Método Dummy Variables
# Coluna 0: Female
# Coluna 1: Infant
# Coluna 2: Male

# Criando a Instância
enc = OneHotEncoder()

# Gerando uma nova matriz(X1) apenas com as Dummy Variables
# '0:1'= Pega a Coluna 0 e 1 para gerar as Dummy Variables
# A função espera receber um array 2D e não apenas 1D
# Acredito que a segunda coluna sirva para a ordenação das Dummy Variables.
X_1 = enc.fit_transform(X[:, 0:1]).toarray()

# Concatenando a Matriz X1 com a matriz original
# Não será utilizada a linha 0 de X pois lá é onde estão as...
# ...variáveis categoricas que precisam ser subtituidas.
# '1:'= seleciona a matriz a partir da coluna 1 e ignora a coluna 0
# 'axis=1'= representa se a concatenação é horizantal ou vertical
X_2 = np.concatenate((X_1, X[:, 1:]), axis=1)
# ========================================================================
# Normalizando os dados

# Utilizada a função StandarScaler do scikit

# Criando Instância
sc_X = StandardScaler()

# Aplicando a função
X_2 = sc_X.fit_transform(X_2)

# ========================================================================
# Implementando Regressão Linear Ordinária

# Dividindo Treino e Teste
# Test_Size = O cojunto de teste será 20% de todos os dados
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_2, Y, test_size=0.2, random_state=0)

# Criando instância modelo
regressor_1 = LinearRegression()

# Treinando o Modelo
regressor_1.fit(X_train_1, y_train_1)

# Testando o modelo
y_pred_1 = regressor_1.predict(X_test_1)

# Avaliação do Modelo
rmse_1 = mean_squared_error(y_test_1, y_pred_1, squared=0)
r2_1 = r2_score(y_test_1, y_pred_1)
print("RMSE da Regrassão Linear Ordinária: ", rmse_1)
print("  R² da Regrassão Linear Ordinária: ", r2_1)

# ========================================================================
# Aplicando KFold-5

# Criando a instância kf
kf = KFold(n_splits=5)

# Criando um loop for que lista todas as opções de folds
for train_index, test_index in kf.split(X_2):
    # Printando o índice das observações utilizadas para teste...
    # ...e para treino para cada diferente fold.
    # print("TRAIN:", train_index, "TEST:", test_index)

    # Alocando o valor dos preditores
    # Acredito que o modelo Preditivo tenha que ser implementado aqui
    X_train_2 = X_2[train_index]
    X_test_2 = X_2[test_index]
    y_train_2 = Y[train_index]
    y_test_2 = Y[test_index]

# Implementando Regressão Linear Para o KFCV5
# Criando instância modelo
    regressor = LinearRegression()

# Treinando o Modelo
    regressor.fit(X_train_2, y_train_2)

# Testando o modelo
    y_pred_2 = regressor.predict(X_test_2)

# Avaliação do Modelo
    rmse_2 = mean_squared_error(y_test_2, y_pred_2, squared=0)
    r2_2 = r2_score(y_test_2, y_pred_2)
    print("RMSE da Regrassão Linear 5-Fold CV: ", rmse_2)
    print("  R² da Regrassão Linear 5-Fold CV: ", r2_2)
