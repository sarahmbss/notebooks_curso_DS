import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

#importação da base
df = pd.read_csv('Credit.csv')

#formato de matriz
previsores = credito.iloc[:,0:20].values
classe = credito.iloc[:,20].values

#transformação dos atributos categóricos em atributos numéricos, passando o índice de cada coluna categórica
#precisamos criar um objeto para cada atributo, pois se forem diferentes, o numero atribuido a cada valor poderá ser diferente,
#gerando inconsistência

#fazer esse código para cada uma das colunas categóricas
labelencoder1 = LabelEncoder()
previsores[:,0] - labelencoder1.fit_transform(previsores[:,0])

#divisão da base de dados entre treinamento e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size = 0.3, random_state=0)

#criação do modelo, treinamento, obtenção das previsões e taxa de acerto
svm = SVC()
svm.fit(X_treinamento, y_treinamento)

#previsoes
previsoes = svm.predict(X_teste)
previsoes

#acerto total do modelo
taxa_acerto = accuracy_score(y_teste, previsoes)

#utilização do algoritmo extratreeclassifier para extrair as características mais importantes
forest = ExtraTreesClassifier()
forest.fit(X_treinamento, y_treinamento)
importancias = forest.feature_importances_

#depois disso só repetir o modelo com as variáveis de maior importância