import numpy as np

class OrdinaryLS:
    def __init__(self):
        self.params=None


    
    def fit(self,x:np.ndarray,y:np.ndarray):
        y=np.asarray(y).reshape(-1)
        x_b=np.hstack([np.ones((x.shape[0],1)),x])
        self.params=np.linalg.inv((x_b.T @ x_b )) @x_b.T @y


    def get_params(self)->np.ndarray:
        return self.params    
       


    def predict(self, x: np.ndarray)->np.ndarray:
        if self.params is None:
            raise ValueError("Modelo não foi treinado. Chame fit() antes de predict().")
        x = np.atleast_2d(x)
        x_b = np.hstack([np.ones((x.shape[0],1)), x])
        return x_b @ self.params
     
     

 
      




        