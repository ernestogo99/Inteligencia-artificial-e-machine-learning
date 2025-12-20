import numpy as np
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

class Kmedias:
    """
    cria um objeto da classe k medias
    """
    def __init__(self, k=5, distance_metric=None):
        self.k = k
        self.distance_metric = distance_metric  
        self.items_centroides = None
        self.centroides = None

    # Definições das funções de distância, que podem ser usadas de fora da classe
    @staticmethod
    def euclidian_distance(a, b, cov_matrix=None):
        return np.linalg.norm(a - b)

    @staticmethod
    def mahalanobis_distance(a, b, cov_matrix):
        delta = a - b
        return np.sqrt(np.dot(np.dot(delta.T, np.linalg.inv(cov_matrix)), delta))


    def initialize_centroids(self, train_data):
        """
        a funcao desse metodo eh inicializar os centroides usando a filosofia do kmedias ++
        """

        array_train_data = np.array(train_data)
        centroides = [array_train_data[np.random.choice(array_train_data.shape[0])]]

        for _ in range(1, self.k):
            distances = np.array([min([np.linalg.norm(point - centroide) for centroide in centroides]) for point in array_train_data])
            squared_distances = distances ** 2
            probabilities = squared_distances / np.sum(squared_distances)
            next_centroid_idx = np.random.choice(array_train_data.shape[0], p=probabilities)
            centroides.append(array_train_data[next_centroid_idx])

        self.centroides = np.array(centroides)
        return self.centroides

    def encontra_particao(self, data):
        """
        funcao para encontrar as particoes de cada instancia do conjunto de dados
        """
        centroides = self.centroides
        array_data = np.array(data)
        
        cov_matrix = np.cov(array_data.T) 
        
        items_centroides = []
        for item in array_data:
            distancias = [self.distance_metric(item, centroide, cov_matrix) for centroide in centroides]
            indice_mais_proximo = np.argmin(distancias)
            items_centroides.append(indice_mais_proximo)
        
        self.items_centroides = items_centroides
        return items_centroides

    def recalcula_centroides(self, data):
        """"
        funcao para recalcular os centroides apos as novas particoes serem encontradas
        """
        array_data = np.array(data)
        new_centroids = []

        for i in range(self.k):
            cluster_items = array_data[np.array(self.items_centroides) == i]
            if len(cluster_items) > 0:
                new_centroid = np.mean(cluster_items, axis=0)
            else:
                new_centroid = self.centroides[i]
            
            new_centroids.append(new_centroid)
        
        return np.array(new_centroids)

    def fit(self, train_data, threshold=10e-6): 
        self.centroides = self.initialize_centroids(train_data)
        centroides_antigos = self.centroides
        self.encontra_particao(train_data)
        diferenca_centroides = np.array([10e6] * self.k)

        while np.max(np.abs(diferenca_centroides)) > threshold:
            novos_centroides = self.recalcula_centroides(train_data)
            diferenca_centroides = novos_centroides - centroides_antigos
            centroides_antigos = novos_centroides
            self.encontra_particao(train_data)
        
        self.centroides = centroides_antigos
        return self.centroides
    


def find_best_k(data, k_values, n_runs=10, distance_metric=None):
    """
    funcao para achar o melhor k
    """
    best_k = None
    best_db_index = float('inf')
    best_labels = None
    best_centroids = None


    for k in k_values:
        best_db_for_k = float('inf')
        best_labels_for_k = None
        best_centroids_for_k = None


        for _ in range(n_runs):
            kmeans = Kmedias(k=k, distance_metric=distance_metric) 
            kmeans.fit(data)
            labels = kmeans.items_centroides
            centroids = kmeans.centroides
            
        
            db_index = davies_bouldin_score(data, labels)
            
            if db_index < best_db_for_k:
                best_db_for_k = db_index
                best_labels_for_k = labels
                best_centroids_for_k = centroids

        if best_db_for_k < best_db_index:
            best_db_index = best_db_for_k
            best_k = k
            best_labels = best_labels_for_k
            best_centroids = best_centroids_for_k

    return best_k, best_labels, best_centroids

def plot_clusters(data, labels, centroids):
    
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centróides')
    plt.title(f'Melhor agrupamento com K={len(centroids)} baseado no Índice Davies-Bouldin')
    plt.legend()
    plt.show()