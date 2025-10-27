import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        X=np.array(X,dtype=float)
        y = y.ravel()
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)

    def predict(self, X):
        X=np.array(X,dtype=float)
        preds = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                prob = -0.5 * np.sum(np.log(2 * np.pi * self.vars[c]))
                prob -= 0.5 * np.sum(((x - self.means[c]) ** 2) / self.vars[c])
                posteriors.append(prior + prob)
            preds.append(np.argmax(posteriors))
        return np.array(preds)


class GaussianDiscriminantAnalysis:
    def fit(self, X, y):
        X=np.array(X,dtype=float)
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.means = {}
        self.priors = {}
        cov = np.zeros((n_features, n_features))

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = len(X_c) / len(X)
            cov += np.cov(X_c, rowvar=False) * len(X_c)

        self.cov = cov / len(X)
        self.cov_inv = np.linalg.inv(self.cov)

    def predict(self, X):
        X=np.array(X,dtype=float)
        preds = []
        for x in X:
            scores = []
            for c in self.classes:
                mean = self.means[c]
                prior = np.log(self.priors[c])
                term = -0.5 * (x - mean).T @ self.cov_inv @ (x - mean)
                scores.append(prior + term)
            preds.append(np.argmax(scores))
        return np.array(preds)
