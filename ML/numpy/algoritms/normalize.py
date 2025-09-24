import numpy as np
class Normalize:
    def __init__(self,dataset:np.ndarray):
        self.dataset=dataset

    
    def length(self):
        return len(self.dataset)


    def mean(self):
        return self.dataset.mean(axis=0)


    def std(self):
        dataset_str=np.sqrt((1/(self.length()-1)) * np.sum((self.dataset -self.mean())**2,axis=0)) 
        return dataset_str
    

    def normalize(self):
        dataset_norm=(self.dataset - self.mean())/self.std()
        return dataset_norm
    
    def desnormalize(self):
        dataset_norm=self.normalize()
        dataset_original=(dataset_norm * self.std()) + self.mean() 
        return dataset_original
    

    

    

