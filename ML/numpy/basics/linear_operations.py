import numpy as np


A = np.array([[2,0],[4,6],[8,2]])

print('A * 5')
print(A*5)
print('A+A')
print(A + A)

#Produto com broadcast(difusão de um vetor por uma matriz)

v1=np.array([5,3])
print('Produto com broadcast')
print(v1 *A)

#Soma com broadcast(difusão) de um vetor por uma matriz
print('soma com broadcast')
print(v1 + A)


#cuidado com as dimensões
v2 = np.array([9,2,1])
print("Vetor v2:")
print(v2, "dimensão:", v2.shape)
print("Matriz A:")
print(A, "dimensão:", A.shape)
try:
    v2 + A
except Exception as e:
    print(e)