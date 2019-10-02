import csv

data = []
with open("house/house.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = (float(row["loyer"]), float(row["surface"]))
        data.append(t)

standard_dev = 2199


import numpy as np

surfaces = np.array([d[1] for d in data])
loyers = np.array([d[0] for d in data])

print(surfaces.shape)
print(loyers.shape)

x = np.matrix([np.ones(surfaces.shape[0]), surfaces]).T
y = np.matrix(loyers).T
theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
print(theta) # 41,-283

import matplotlib.pyplot as plt

plt.plot(surfaces, loyers, 'ro', markersize=4)
plt.plot(range(400), [41*x-283 for x in range(400)] )
plt.show()


