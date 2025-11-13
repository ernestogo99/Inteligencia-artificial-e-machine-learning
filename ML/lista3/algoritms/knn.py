import numpy as np

class KNNClassifier:
    def __init__(self, k=3, metric='euclidean', V=None):
        """
        k: número de vizinhos
        metric: 'euclidean' ou 'mahalanobis'
        V: matriz de covariância (usada só para Mahalanobis)
        """
        self.k = k
        self.metric = metric
        self.V = V

    def fit(self, X, y):    
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        if self.metric == 'mahalanobis' and self.V is None:
            self.V = np.cov(self.X_train, rowvar=False)
            self.V_inv = np.linalg.inv(self.V)
        elif self.metric == 'mahalanobis':
            self.V_inv = np.linalg.inv(self.V)

    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.linalg.norm(x1-x2)
        elif self.metric == 'mahalanobis':
            diff = x1 - x2
            return np.sqrt(diff.T @ self.V_inv @ diff)
        else:
            raise ValueError("Métrica desconhecida: use 'euclidean' ou 'mahalanobis'.")

    def predict(self, X)->np.ndarray:
        X = np.array(X, dtype=float)
        preds = []

        for x in X:
  
            distances = np.array([self._distance(x, x_train) for x_train in self.X_train])
            k_indices = distances.argsort()[:self.k]
            k_labels = self.y_train[k_indices]
            values, counts = np.unique(k_labels, return_counts=True)
            preds.append(values[np.argmax(counts)])
        return np.array(preds)
