from sklearn import datasets
from sklearn.metrics import confusion_matrix
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

iris = datasets.load_iris

#configuração dos parâmetros do k-medoids
#3, 12 e 20 são índices aleatórios de registros da base de dados (inicialização)
cluster = kmedoids(iris.data[:,0:2], [3, 12, 20])

#visualização dos pontos escolhidos (3, 12, 20)
cluster.get_medoids()

#aplicação do algoritmo para o agrupamento, obtenção das previsões (grupo de cada registro) e visualização dos medoides
cluster.process()
previsoes = cluster.get_clusters()
medoides = cluster.get_medoids()

#visualização do agrupamento
v = cluster_visualizer()
v.append_clusters(previsoes, iris.data[:,0:2])
v.append_cluster(medoides, data=iris.data[:,0:2], marker='*', markersize=20)
v.show()

#código para criar 2 listas, uma com os grupos reais da base de dados e outra com os valores dos grupos

lista_previsoes = []
lista_real = []

for i in range(len(previsoes)):
    for j in range(len(previsoes[i])):
        lista_previsoes.append(i)
        lista_real.append(iris.target[previsoes[i][j]])

#geração da matriz de contingência, comparando os grupos reais com os grupos previstos
lista_previsoes = np.asarray(lista_previsoes)
lista_real = np.asarray(lista_real)
resultados = confusion_matrix(lista_real, lista_previsoes)
resultados