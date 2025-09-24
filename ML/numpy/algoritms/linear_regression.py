import numpy as np

class LinearRegression:
    def __init__(self,alpha:float,iterations:int):
        self.alpha=alpha
        self.iterations=iterations
        self.loss_history=[]
        self.w=None



    def fit(self,x:np.ndarray,y:np.ndarray)->dict:
        N=x.shape[0]
        D=x.shape[1]
        x=np.hstack([np.ones((N,1)),x])
        y=np.asarray(y).reshape(-1)
        self.loss_history=[]
        self.w=np.zeros((D+1,1)).reshape(-1)

        for _ in range(self.iterations):
            y_hat= x @ self.w
            error = y-y_hat

            self.w+= (self.alpha/N) * (error @ x)
            rmse =np.sqrt((np.mean((error) ** 2)))
            self.loss_history.append(rmse)

        return {'w':self.w,'error':np.array(self.loss_history)}
    

    def fit_stochastic_descent(self,x:np.ndarray,y:np.ndarray)->dict:
        N=x.shape[0]
        D=x.shape[1]
        x=np.hstack([np.ones((N,1)),x])
        y=np.asarray(y).reshape(-1)
        self.w=np.zeros((D+1,1)).reshape(-1)
        self.loss_history=[]
        random_data=np.random.default_rng()
        for _ in range(self.iterations):
            idx=random_data.permutation(N)

            for i in idx:
                x_i = x[i]
                y_hat=x_i @ self.w
                error =y[i]-y_hat
                self.w+= self.alpha * error * x_i


            y_hat_all= x @ self.w    
            rmse =np.sqrt((np.mean((y-y_hat_all) ** 2)))
            self.loss_history.append(rmse)
        return {'w':self.w,'error':np.array(self.loss_history)}
           

           
                
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        N = x.shape[0]
        x_b = np.hstack([np.ones((N, 1)), x])  
        return x_b @ self.w

                



        

        