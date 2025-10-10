import numpy as np

class Metrics:
    def __init__(self):
        pass


    def mse(self,y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def rmse(self,y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(self.mse(y_true, y_pred))