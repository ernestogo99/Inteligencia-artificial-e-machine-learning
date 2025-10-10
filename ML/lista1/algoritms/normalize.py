import numpy as np
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X:np.ndarray)->np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        return (X_norm * self.std_) + self.mean_



class MinMaxScaler:
    def __init__(self):
        self.min_=None
        self.max_=None
        
    def fit(self, X):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self
    
    def transform(self, X)->np.ndarray:
        return (X - self.min_) / (self.max_ - self.min_)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_norm):
        return X_norm * (self.max_ - self.min_) + self.min_

    