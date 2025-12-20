from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from models.kmedia import Kmedias, find_best_k, plot_clusters
from models.pca import PCA


def main_q1():
    dados_1 = pd.read_csv('data/quake.csv')
    dados_1 = dados_1.sample(frac= 1).reset_index(drop= True)
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados_1.select_dtypes(include=[np.number]))

    k_values = range(4, 21)

    best_k, best_labels, best_centroids = find_best_k(dados_normalizados, k_values=k_values, n_runs=10, distance_metric=Kmedias.euclidian_distance)

    plot_clusters(dados_normalizados, best_labels, best_centroids)



def main_q1_b():
    dados_1 = pd.read_csv('data/quake.csv')
    dados_1 = dados_1.sample(frac= 1).reset_index(drop= True)
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados_1.select_dtypes(include=[np.number]))

    k_values = range(4, 21)
    best_k, best_labels, best_centroids = find_best_k(dados_normalizados, k_values=k_values, n_runs=10, distance_metric=Kmedias.mahalanobis_distance)

    plot_clusters(dados_normalizados, best_labels, best_centroids)

def main_q2():
    dados_2 = pd.read_csv('data/penguins.csv')
    dados_2.sample(frac=1).reset_index(drop =True)
    scaler = StandardScaler()
    dados_normalizados_2 = scaler.fit_transform(dados_2.select_dtypes(include=[np.number]))
    pca = PCA(n_componentes=2)
    data_projetado = pca.fit_transform(data= dados_normalizados_2)

    plt.figure(figsize=(8, 6))
    plt.scatter(data_projetado[:, 0], data_projetado[:, 1], c='blue', label='Dados projetados')
    plt.title('Projeção dos Dados após PCA em 2 Componentes')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(1, 5):
        print('='*30)
        print(f'Usando a dimensão {i}')
        pca = PCA(n_componentes= i)
        res = pca.fit_transform(dados_normalizados_2)
        print(f'Variancia explicada: {pca.variancia_explicada}')
if __name__ =='__main__':
    main_q2()