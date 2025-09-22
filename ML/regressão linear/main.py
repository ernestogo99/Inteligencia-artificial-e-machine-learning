import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



base_dados=pd.read_excel('BaseDados_RegressaoLinear.xlsx','Plan1') # nome do arquivo e sua aba
print(base_dados)
base_dados.head() # mostra os 5 primeiros registros, posso colocar parametros
base_dados.tail() # mostra os ultimos 5 registros , posso colocar parametros
base_dados.columns # mostra o nome das colunas
base_dados.info() # mostra as colunas, valores e o formato dos campos
base_dados.describe() # gera uma tabela estatistica com base nos dados
print(base_dados.columns)
print(base_dados.describe())

# converter para um array
# ganha mais performance no desenvolvimento
# o método iloc nos permiter quebrar os dados, primeiro eu passo quantas informações eu quero e depois a coluna
eixo_x=base_dados.iloc[:,0].values
eixo_y=base_dados.iloc[:,1].values

for i in range(len(eixo_x)):
    print(f'posição:{i}, valor:{eixo_x[i]}' )


# visualização gráfica

plt.figure(figsize=(10,5)) # horizontal e vertical
#plt.scatter(eixo_x,eixo_y) # gerar o gráfico 
plt.title('Gráfico com dois eixos [salario x limite]')
plt.xlabel('Salário')
plt.ylabel('Limite')
#plt.show()

#sbn.heatmap(base_dados.isnull()) # mapa de calor, para ver valores nulos
#sbn.pairplot(base_dados) # pega a base de dados e faz um gráfico de dois eixos para todas as variaveis


# correlação (ela avalia a relação entre as variáveis)

correlacao=np.corrcoef(eixo_x,eixo_y)
#sbn.heatmap(correlacao,annot=True)
#plt.show()
print(correlacao)

# modelo 

# converter os valores para a forma de matriz
eixo_x=eixo_x.reshape(-1,1)
eixo_y=eixo_y.reshape(-1,1)

# dividir os dados em treino e teste

x_treinamento,x_teste,y_treinamento,y_teste=train_test_split(
    eixo_x,eixo_y,test_size=0.20
)

print(len(x_treinamento))

# regressão linear
regressao=LinearRegression()

regressao.fit(x_treinamento,y_treinamento) # aplica os calculos estatisticos nos dados para treinar o modelo

score=regressao.score(x_treinamento,y_treinamento) # calcula o quanto nosso modelo se aproximou da correlação(o quanto as variaveis se explicaram)
print(score)
plt.scatter(x_treinamento,y_treinamento)
plt.plot(x_teste,regressao.predict(x_teste),color='red')
plt.show() 

# avaliando o desempenho do modelo
previsoes=regressao.predict(x_teste)

print('RMSE', np.sqrt(metrics.mean_squared_error(y_teste,previsoes))) # avaliando se o y teste ficou perto das previsoes

# exemplo (pessoa que ganha 1800 reais, terá quantos reais de limite?)
print(regressao.predict([[1800]]))
