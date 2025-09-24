import numpy as np


pressao_dataset=np.genfromtxt('../data/pressão.txt',delimiter=',',skip_header=1)
print('dataset de pressão')
print(pressao_dataset)


peixe_dataset=np.genfromtxt('../data/peixe.txt',delimiter=',',skip_header=1)
print('dataset de peixes')
print(peixe_dataset)