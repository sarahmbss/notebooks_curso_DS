from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#carregamento da base
iris = datasets.load_iris

#visualização de quantos registros existem por classe
unicos, quantidade = np.unique(iris.target, return_counts=True)

cluster = KMeans(n_clusters=3)
cluster.fit(iris.data)

centroides = cluster.cluster_centers_

previsoes = cluster.labels_

#contagem dos registros por classe
unicos2, quantidade2 = np.unique(previsoes, return_counts=True) #aqui fica claro que a classificação para os grupos 2 e 3
#ficaram diferentes do real

#geração do gráfico com os clusters gerados
plt.scatter(iris.data[previsoes == 0,0], iris.data[previsoes==0, 1], c='green', label='Setosa')
plt.scatter(iris.data[previsoes == 1,0], iris.data[previsoes==1, 1], c='red', label='Versicolor')
plt.scatter(iris.data[previsoes == 2,0], iris.data[previsoes==2, 1], c='blue', label='Virgica')
plt.legend() 

