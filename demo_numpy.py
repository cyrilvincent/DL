import numpy as np

a1 = np.array([1,5,6,7])
a2 = np.arange(4) #[0,1,2,3]
print(a1 + a2)
print(a1 * 2)
print(np.sin(a1))

m1 = np.array([[1,2],[3,4]])
print(m1)
print(np.sin(m1))

print(m1.shape)
print(a1.shape)

a3 = np.array([1,2,3,4,5,6,7,8,9])
print(a3.shape)
m3 = a3.reshape(3,3)
print(m3)

print(m3[1,2])
print(m3[1:,:2])