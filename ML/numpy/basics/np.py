import numpy as np

A = np.array([[2,0],[4,6],[8,2]])
B = np.array([[1,3],[5,7]])
v1 = np.array([5,3])
v2 = np.array([9,2,1])

print(f'A: {A}')
print(f'B: {B}')
print(f'v1: {v1}')
print(f'v2:{v2}')

print("Dimensão de A:", A.shape)
print("Dimensão de B:", B.shape)
print("Dimensão de v1:", v1.shape)
print("Dimensão de v2:", v1.shape)


print("Matriz original:")
print(A)
print("Matriz transposta:")
print(A.T)

print("Primeira linha de A:")
print(A[0], "dimensão:", A[0].shape) # Primeira linha de A
print("Primeira e terceira linha de A:")
print(A[[0,2]], "dimensão:", A[[0,2]].shape) # Primeira e terceira linha de A
print("Primeira linha de A (mantendo 2 dimensões):")
print(A[[0]], "dimensão:", A[[0]].shape) # Primeira linha de A (mantendo 2 dimensões)

a = np.array([2, 4, 8]).reshape((1,-1))
b = np.array([1, 2, 3]).reshape((1,-1))
ab = np.r_[a,b]

print(a, "dimensão:", a.shape)
print(b, "dimensão:", b.shape)
print(ab, "dimensão:", ab.shape) # row stack

c = np.array([2, 4, 8]).reshape((-1,1))
d = np.array([1, 2, 3]).reshape((-1,1))
cd = np.c_[c,d]

print(c, "dimensão:", c.shape)
print(d, "dimensão:", d.shape)
print(cd, "dimensão:", cd.shape) # column stack