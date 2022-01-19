from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
from scipy import stats

iris = datasets.load_iris()
stats.describe(iris.data)

#criação dos previsores
previsores = iris.data
classe = iris.target

#divisão da base em treinamento, teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size = 0.3, random_state=0)

#criação do modelo
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_treinamento, y_treinamento)

#obtenção das previsões
previsoes = knn.predict(X_teste)
previsoes

#matriz de confusão
confusao = confusion_matrix(y_teste, previsoes)

#acerto total do modelo
taxa_acerto = accuracy_score(y_teste, previsoes)