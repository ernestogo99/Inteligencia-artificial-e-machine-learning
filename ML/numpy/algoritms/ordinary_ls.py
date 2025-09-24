import numpy as np

class OrdinaryLS:
    def __init__(self):
        self.params=None



    def fit(self,x:np.ndarray,y:np.ndarray):
        y=np.asarray(y).reshape(-1)
        x_b=np.hstack([np.ones((x.shape[0],1)),x])
        self.params=np.linalg.inv((x_b.T @ x_b )) @x_b.T @y
       


    def predict(self, x: np.ndarray):
        if self.params is None:
            raise ValueError("Modelo não foi treinado. Chame fit() antes de predict().")
        x = np.atleast_2d(x)
        x_b = np.hstack([np.ones((x.shape[0],1)), x])
        return x_b @ self.params
     
    

    def rmse(self,x:np.ndarray,y:np.ndarray):
        y_pred=self.predict(x)
        y=np.asarray(y).reshape(-1)
        rmse=np.sqrt(np.mean((y-y_pred)**2))
        return rmse
      




        