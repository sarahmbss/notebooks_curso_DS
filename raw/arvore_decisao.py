import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

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

#criaçao e treinamento do modelo
arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento, y_treinamento)

#exportação da árvore de decisão para o formato .dot
export_graphviz(arvore, out_file='tree.dot') #quando joga para o site webgraphviz.com ele gera a estrutura visual da árvore

#obteção das previsões
previsoes = arvore.predict(X_teste)

#matriz de confusão
confusao = confusion_matrix(y_teste, previsoes)

#taxa de acerto
taxa_acerto = accuracy_score(y_teste, previsoes)