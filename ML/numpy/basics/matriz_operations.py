import numpy as np

A = np.array([[2,0],[4,6],[8,2]])
B = np.array([[1,3],[5,7]])
v1 = np.array([5,3])
print('multiplicação de matrizes no numpy')
print(A @ B)
print("Dimensão de A @ B:", (A@B).shape)


# cuidado com as dimensões

try:
    B @ A
except Exception as e:
    print(e)


b_inv=np.linalg.inv(B)
print('B')
print(B)
print('Inversa de B')
print(b_inv)
print('B * B ^-1')
print(b_inv @ B)

print(' A inversa pode ser usada no produto de vetores')
print(np.linalg.solve(B,v1))