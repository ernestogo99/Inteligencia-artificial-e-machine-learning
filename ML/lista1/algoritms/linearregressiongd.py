import numpy as np
from .metrics import Metrics

class LinearRegressionGD:
    def __init__(self, alpha: float = 0.01, iterations: int = 1000):
        self.alpha = alpha
        self.iterations = iterations
        self.loss_history = []
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        metrics=Metrics()
        N, D = X.shape
        X = np.hstack([np.ones((N, 1)), X])  
        y = y.reshape(-1)

        self.w = np.zeros(D + 1)  

        for _ in range(self.iterations):
            y_hat = X @ self.w
            error = y - y_hat

           
            self.w += (self.alpha / N) * (X.T @ error)

  
            self.loss_history.append(metrics.rmse(y_true=y, y_pred=y_hat))

        return self  
    

    def get_params(self)->np.ndarray:
        return self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X = np.hstack([np.ones((N, 1)), X])
        return X @ self.w
    

    def get_loss_history(self)->np.ndarray:
        return np.array(self.loss_history)



