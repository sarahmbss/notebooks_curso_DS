#semelhante a linear, mas a variavel de resposta é binária: sucesso ou fracasso
#classificação feita de acordo com probabilidades
#valores abaixo de 0.5 pertencem ao fracasso e acima ao sucesso. podemos olhar também como a possibilidade de sucesso

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

np.corrcoef(base.DESPESAS, base.SITUACAO) #verifica a correlação entre as variáveis

#criação das variaveis dependentes e independentes
#transformação do X para o formato de matriz adicionando um novo eixo (newaxis)
x = base.iloc[:, 2].values
x = x[:, np.newaxis]
y = base.iloc[:, 1]. values

#criação do modelo
modelo = LogisticRegression()
modelo.fit(X, y)
modelo.coef_
modelo.intercept_

#previsao dos novos candidatos
despesas = base_previsoes.iloc[:, 1].values
despesas = despesas.reshape(-1, 1)
previsoes_teste = modelo.predict(despesas)

#adicionando a nova coluna com a previsão com a base
base_previsoes = no.column_stack((base_previsoes, previsoes_teste))
