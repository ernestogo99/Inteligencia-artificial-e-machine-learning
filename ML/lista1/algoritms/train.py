import numpy as np
def train_test_split(x:np.ndarray,y:np.ndarray,test_size:float,random_state:int=None):
    n_samples=x.shape[0]
    n_test=int(n_samples* test_size)

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_indices=indices[:n_test]
    train_indices=indices[n_test:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test
    
