import numpy as np
from algoritms.normalize import Normalize
from algoritms.newton_raphson import newtow_raphson
from algoritms.ordinary_ls import OrdinaryLS
def main():
    #Exercício 1
    data_peixes=np.genfromtxt('data/peixe.txt',delimiter=',',skip_header=1)
    normalize=Normalize(data_peixes)
    print(normalize.normalize())

    #Exercício 2
    f= lambda x: x**2 -2
    f_prime= lambda x:2*x
    x0 = 1
    raiz = newtow_raphson(f, f_prime, x0)
    print("Raiz aproximada:", raiz)
    print("Verificação: raiz^2 =", raiz**2) 

    #Exercício 3
    ordinary_ls=OrdinaryLS()
    test_x=data_peixes[:,[0,1]]
    y=data_peixes[:,[2]]
    ordinary_ls.fit(test_x,y)
    print('error')
    print(ordinary_ls.rmse(test_x,y))
    print('teste')
    x_teste=np.array([100,27])
    print(ordinary_ls.predict(x_teste))
 




if __name__ == "__main__":
    main()