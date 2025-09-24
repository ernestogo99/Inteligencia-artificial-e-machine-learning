import numpy as np

print("Matriz identidade de ordem 5:")
print(np.eye(5)) # np.eye cria matriz identidade
print("Matriz de zeros 5x3:")
print(np.zeros((5,3))) # np.zeros cria uma matriz de zeros
print("Matriz de uns 3x1:")
print(np.ones((3,1))) #np.ones cria uma matriz de uns

# matrizes aleatorias

print("Matriz 5x3 de números aleatórios amostrados de U(0,1):")
print(np.random.rand(5,3))
print("Matriz 2x2 de números aleatórios amostrados de N(0,1):")
print(np.random.randn(2,2))