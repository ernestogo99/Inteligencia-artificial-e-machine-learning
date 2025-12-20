import numpy as np

class PCA():
    """
    cria um objeto da classe PCA
    """
    def __init__(self, n_componentes):
        self.n_componentes = n_componentes
        self.matriz_projecao = None
        self.variancia_explicada = None

    def fit_transform(self, data):
        """
        faz o fit do modelo aos dados e retorna eles projetados na dimensao escolhida
        """
        m = self.n_componentes
        data_array = np.array(data)
        
    
        cov_matrix = np.cov(data_array.T)
        
    
        autovalores, autovetores = np.linalg.eigh(cov_matrix)
        
      
        index = np.argsort(autovalores)[::-1]  
        autovalores = autovalores[index]  
        autovetores = autovetores[:, index]  
        
      
        autovetores_selecionados = autovetores[:, :self.n_componentes]
        
        
        self.matriz_projecao = autovetores_selecionados
        
        self.variancia_explicada = np.sum(autovalores[:self.n_componentes]) / np.sum(autovalores)

  
        return np.dot(data_array, self.matriz_projecao)