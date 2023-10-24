import numpy as np

v1 = np.array([1,2,3,4])
print(v1)
v2 = np.arange(4)
print(v2)
print(v1 * v2)
print(v1.shape)

mat1 = np.array([[1,2,3],[4,5,6]])
print(mat1)
print(mat1.T)
# Row first
print(mat1.shape)
print(f"Rows: {mat1.shape[0]}, Columns: {mat1.shape[1]}")