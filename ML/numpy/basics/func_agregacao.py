import numpy as np


A = np.array([[2,0],[4,6],[8,2]])


print(A.mean()) # média da matriz
print(A.sum()) # soma da matriz
print(A.prod()) # produto da matriz

print('axis =0')
print(A.mean(axis=0)) # axis=0 pega oq eu quero tirar no caso a coluna, a primeira dimensão do array
print(A.sum(axis=0))
print(A.prod(axis=0))

print('axis = 1')
print(A.mean(axis=1)) # axis =1 pega a linha
print(A.sum(axis=1))
print(A.prod(axis=1))

print('keepdins')
print(A.mean(axis=0, keepdims=True))