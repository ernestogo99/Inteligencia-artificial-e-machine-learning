import numpy as np

class Metrics:
    def __init__(self):
        pass


    def mse(self,y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def rmse(self,y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(self.mse(y_true, y_pred))
    
    def polynomial_features(self,X: np.ndarray, degree: int) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.hstack([X**i for i in range(1, degree + 1)])