import numpy as np

class RidgeRegression:
    def __init__(self, lamda: float = 0.01):
        self.lamda = lamda
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        X = np.hstack([np.ones((N, 1)), X])  
        D = X.shape[1]
        I = np.eye(D)
        self.w = np.linalg.inv(X.T @ X + self.lamda * I) @ X.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X = np.hstack([np.ones((N, 1)), X])
        return X @ self.w
