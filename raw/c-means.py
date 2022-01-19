from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import skfuzzy

'''
Qual a diferença para o kmeans?

O c-means mostra a porcentagem que cada elemento está em cada grupo diferente. 
Exemplo: A pertence 70% ao grupo A e 30% ao grupo B
'''

#carregamento da base
iris = datasets.load_iris

#aplicação do algoritmo com 3 clusters
r = skfuzzy.cmeans(data = iris.data.T, c=3, m=2, error = 0.005, maxiter = 1000, init=None)

#obtendo as porcentagens de um registro permanecer a um cluster
previsoes_porcentagem = r[1]

#visualização da probabilidade de um registro pertencer a cada um dos clusters
for x in range(150):
    print(previsoes_porcentagem[0][x], previsoes_porcentagem[1][x], previsoes_porcentagem[2][x])

#geração de matriz de contingência para comparação com as classes originais da base de dados
previsoes = previsoes_porcentagem.argmax(axis=0)
resultados = confusion_matrix(iris.target, previsoes)