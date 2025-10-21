import numpy as np
from .metrics import Metrics

class LinearRegressionSGD:
    def __init__(self,alpha:float=0.01,iterations:int=1000):
        self.alpha=alpha
        self.iterations=iterations
        self.loss_history=[]
        self.w=None
        

    def fit_stochastic_descent(self,x:np.ndarray,y:np.ndarray)->dict:
        metrics=Metrics()
        N=x.shape[0]
        D=x.shape[1]
        x=np.hstack([np.ones((N,1)),x])
        y=np.asarray(y).reshape(-1)
        self.w=np.zeros((D+1,1)).reshape(-1)
        random_data=np.random.default_rng()
        for _ in range(self.iterations):
            idx=random_data.permutation(N)

            for i in idx:
                x_i = x[i]
                y_hat=x_i @ self.w
                error =y[i]-y_hat
                self.w+= self.alpha * error * x_i


            y_hat_all= x @ self.w    
            self.loss_history.append(metrics.rmse(y,y_hat_all))
        return self
    

    def get_params(self)->np.ndarray:
        return self.w

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X = np.hstack([np.ones((N, 1)), X])
        return X @ self.w  
    

    def get_loss_history(self)->np.ndarray:
        return np.array(self.loss_history)
  