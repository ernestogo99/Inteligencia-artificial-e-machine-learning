import numpy as np
from algoritms.normalize import Normalize
from algoritms.newton_raphson import newtow_raphson
from algoritms.ordinary_ls import OrdinaryLS
from algoritms.linear_regression import LinearRegression
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

    #Exercício 4
    linear_gd=LinearRegression(alpha=0.01,iterations=1000)
    normalize_x=Normalize(test_x)
    normalized_x=normalize_x.normalize()
    normalize_y=Normalize(y)
    normalized_y=normalize_y.normalize()
    res=linear_gd.fit(normalized_x,normalized_y)
    #print(res)
    print('teste2')
    print(linear_gd.predict(x_teste))

    #Exercício 5
    linear_gd.fit_stochastic_descent(normalized_x,normalized_y)
    print('teste 3')
    print(linear_gd.predict(x_teste))


 




if __name__ == "__main__":
    main()