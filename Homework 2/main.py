import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#========================================================================================================
#Lendo o data frame

df = pd.read_csv("abalone.csv")

#========================================================================================================
#Selecionando as variáveis independentes X e as variáveis dependentes Y

# ':' = Seleciona todas as linhas
#':8' = Seleciona as colunas do ínicio (coluna 0) até a coluna 8(não inclui a coluna 8)
X = df.iloc[:,:8].values

# ':' = Seleciona todas as linhas
#':8' = Seleciona a coluna 8(Rings)(Foi formando como array vertical. não sei se dá problema)
Y = df.iloc[:,8].values

#========================================================================================================
#Existe um processo para tratar dados faltando mas não acreditor que seja necessário

#========================================================================================================
#Processo de Tratamento da variáveis categoricas (Male, Female e Infant)

#Abordagem utilizada: Método Dummy Variables
#Coluna 0: Female
#Coluna 1: Infant
#Coluna 2: Male

#Criando a Instância
enc = OneHotEncoder()

#Gerando uma nova matriz(X1) apenas com as Dummy Variables
#'0:1'= Pega a Coluna 0 e 1 para gerar as Dummy Variables. A função espera receber um array 2D e não apenas 1D. 
# Acredito que a segunda coluna sirva para a ordenação das Dummy Variables.
X1 = enc.fit_transform(X[:,0:1]).toarray()

#Concatenando a Matriz X1 com a matriz original. Não será utilizada a linha 0 de X pois lá é onde estão as...
#...variáveis categoricas que precisam ser subtituidas.
# '1:'= seleciona a matriz a partir da coluna 1 e ignora a coluna 0.
# 'axis=1'= representa se a concatenação é horizantal ou vertical. 1 para horizontal. 
X2= np.concatenate((X1,X[:,1:]),axis=1)

#========================================================================================================
#Normalizando os dados

#Utilizada a função StandarScaler do scikit

#Criando Instância
sc_X = StandardScaler()

#Aplicando a função
X2= sc_X.fit_transform(X2)