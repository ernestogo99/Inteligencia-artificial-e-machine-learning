import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


base_dados=pd.read_excel('BaseDados_RegressaoLogistica.xlsx','Plan1')
#print(base_dados) # mostra a base de dados
base_dados.head() # mostra os 5 primeiros registros, posso colocar parametros
base_dados.tail() # mostra os ultimos 5 registros , posso colocar parametros
#print(base_dados.describe()) # mostra umas estatisticas sobre a base de dados

sns.set_theme(font_scale=1.3,rc={'figure.figsize':(20,20)})
eixo=base_dados.hist(bins=20,color='blue')

print(eixo)