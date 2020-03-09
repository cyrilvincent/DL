print("Hello World!")

l1 = [1.,3,8,9]
print(max(l1))

import math
print(math.tanh(0))

print(type(l1))

res = [math.tanh(x) for x in l1]
print(res)

f = lambda x : math.sin(x) / x

res = [f(x) for x in l1]
print(res)

import numpy as np
v1 = np.array(l1,dtype=np.float64)
print(v1)
v2 = np.arange(2,6,1)
print(v2)
print(v1 + v2)
print(np.tanh(v1))

g = lambda x : np.sin(x) / x
print(g(v1))

mat1 = np.array([[1,2],[3,4],[5,6]])
print(mat1)
print(mat1.shape)
print(v1.shape)
