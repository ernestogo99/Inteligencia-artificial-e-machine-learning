import numpy as np

class LogisticRegressionGD:
    def __init__(self, alpha=0.01, iterations=1000):
        self.alpha = alpha
        self.iterations = iterations
        self.w = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        N, D = X.shape
        X = np.hstack([np.ones((N, 1)), X])
        y = y.reshape(-1, 1)
        self.w = np.zeros((D + 1, 1))

        for _ in range(self.iterations):
            y_hat = self._sigmoid(X @ self.w)
            error = y_hat - y
            self.w -= (self.alpha / N) * (X.T @ error)

           
            loss = -np.mean(y * np.log(y_hat + 1e-10) + (1 - y) * np.log(1 - y_hat + 1e-10))
            self.loss_history.append(loss)

    def predict_proba(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(X @ self.w)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticRegressionSoftmaxGD:
    def __init__(self, alpha=0.01, iterations=500):
        self.alpha = alpha
        self.iterations = iterations
        self.loss_history = []

    def _softmax(self, Z):
        Z = np.clip(Z, -500, 500)
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e_Z / e_Z.sum(axis=1, keepdims=True)

    def _one_hot(self, y):
        classes = np.unique(y)
        self.classes_ = classes
        y_one_hot = np.zeros((len(y), len(classes)))
        for i, c in enumerate(classes):
            y_one_hot[:, i] = (y == c).astype(float)
        return y_one_hot

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y).ravel()
        y_one_hot = self._one_hot(y)

        n_samples, n_features = X.shape
        n_classes = y_one_hot.shape[1]

  
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        for i in range(self.iterations):
            Z = X @ self.W + self.b  
            Y_hat = self._softmax(Z)

        
            grad_W = (X.T @ (Y_hat - y_one_hot)) / n_samples
            grad_b = np.sum(Y_hat - y_one_hot, axis=0, keepdims=True) / n_samples

   
            self.W -= self.alpha * grad_W
            self.b -= self.alpha * grad_b

    
            loss = -np.sum(y_one_hot * np.log(Y_hat + 1e-15)) / n_samples
            self.loss_history.append(loss)

    def predict(self, X):
        X = np.array(X, dtype=float)
        Z = X @ self.W + self.b
        Y_hat = self._softmax(Z)
        return self.classes_[np.argmax(Y_hat, axis=1)]
