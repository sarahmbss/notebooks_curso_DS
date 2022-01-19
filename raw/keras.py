from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

#carregamento dos dados e criação de previsores
base = datasets.load_iris()
previsores = base.data
classe = base.target

#transformação da classe para o formato dummy pois temos uma rede neural com 3 neurônios na camada de saída
classe_dummy = np_utils.to_categorical(classe)

#divisão da base de dados entre treinamento e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe_dummy, test_size=0.3, random_state=0)

#criação da estrutura da rede neural com a classe sequential
modelo = Sequential()
#primeira camada oculta, cinco neurônios, quatro neuronios de entrada
modelo.add(Dense(units=5, input_dim=4)) #como é a primeira precisa do input_dim, quatro pois são as variaveis
#segunda camada oculta
modelo.add(Dense(units=4))
#terceira camada oculta
#softmax pq temos um problema de classificação com mais de 2 classes (é gerada uma probabilidade em cada neurônio)
modelo.add(Dense(units=3, activation='softmax')) #softmax gera probabilidade

#visualização da estrutura da rede neural
modelo.summary()

#configuração dos parâmetros da rede neural
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#treinamento, dividindo a base de treinamento em uma porção para validação (validation_data)
modelo.fit(X_treinamento, y_treinamento, epochs=1000, validation_data=(X_teste, y_teste))

#previsões e mudar a variável para True ou False de acodo com o threshold 0.5
previsoes = modelo.predict(X_teste)
previsoes = (previsoes>0.5)

#como é um problema com três saída, precisamos buscar a posição que possui o maior valor (são retornados 3 valores)
y_teste_matrix = [np.argmax(t) for t in y_teste]
y_previsao_matrix = [np.argmax(t) for t in previsoes]

#geração da matriz de confusão
confusao = confusion_matrix(y_teste_matrix, y_previsao_matrix)