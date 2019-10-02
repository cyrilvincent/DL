import csv

data = []
with open("house/house.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = (float(row["loyer"]), float(row["surface"]))
        data.append(t)

standard_dev = 2199
surface_max = 300
data = [d for d in data if d[1] <= surface_max and abs(41 * d[1] - 283 - d[0]) < 3 * standard_dev]

import numpy as np
surfaces = np.array([d[1] for d in data])
loyers = np.array([d[0] for d in data])

import scipy.stats
theta = scipy.stats.linregress(surfaces, loyers)
print(theta) # 32,178

loyers_predict = np.array([theta.slope*s + theta.intercept for s in surfaces])

import matplotlib.pyplot as plt

plt.plot(surfaces, loyers, 'ro', markersize=4)
plt.plot(range(400), [31*x+206 for x in range(400)] )
plt.show()
