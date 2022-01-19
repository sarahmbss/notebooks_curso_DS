import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix

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

#criação e treinamento do modelo
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

#previsões utilizando os registros de teste
previsoes = naive_bayes.predict(X_teste)

#geração da matriz de confusão e cálculo da taxa de acerto e erro. é aquela que mostra os verdadeiros e falsos positivos
confusao = confusion_matrix(y_teste, previsoes)

#verificando a taxa de acerto
taxa_acerto = accuracy_score(y_teste, previsoes)

#Visualização mais bonita da matriz de confusão utilizando o yellowbrick
v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.proof()

#previsão com novo registro
novo_df = pd.read_csv('NovoCredit.csv')

#usar o mesmo objeto para transformar as variáveis
novo_df = novo_df.iloc[:,0:20].values
novo_df[:0] = labelencoder1.transform(novo_df[:,0]) #fazer isso para todas as outras variáveis que foram transformadas

naive_bayes.predict(novo_df) #previsão do modelo