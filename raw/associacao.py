import pandas as pd
from apyori import apriori

#leitura das transações
dados = pd.read_csv('transacoes.txt', header=None)

#transformação para o formato de lista que é exigido pela biblioteca apyori - 6 é a quantidade de itens na base
transacoes=[]
for i in range(0,6):
    transacoes.append([str(dados.values[i, j] for j in range(0,3))])

#execução do algoritmo apriori para geração das regras de associação definindo os parâmetros de suporte e confiança
regras = apriori(transacoes, min_support=0.5, min_confidence=0.5, min_length=2)

#criação de nova variável para armazenar somente as regras de associação
resultados = list(regras)

#criação de uma nova variável percorrendo a variável anterior para uma melhor visualização dos dados
resultados2 = [list(x) for x in resultados]

#criação de outra variável para a visualização das regras ficar mais fácil para o usuário, adicionando as regras encontradas na variável resultados2
resultados3=[]
for j in range(0,7):
    resultados3.append([list(x) for x in resultados2[j][2]])