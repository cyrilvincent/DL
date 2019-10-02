import csv
data = []
with open("house/house.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = (float(row["loyer"]), float(row["surface"]))
        data.append(t)
print(data)
import numpy as np
surfaces = np.array([d[1] for d in data])
loyers = np.array([d[0] for d in data])
mean = np.mean(loyers)
standard_dev = np.std(loyers)
print(f"Mean: {mean}, Std: {standard_dev}")
import scipy.stats
theta = scipy.stats.linregress(surfaces, loyers)
print(theta)
loyers_predict = np.array([theta.slope*s + theta.intercept for s in surfaces])
import matplotlib.pyplot as plt
plt.plot(surfaces, loyers, 'ro', markersize=4)
plt.plot(range(400), [theta.slope*x + theta.intercept for x in range(400)] )
plt.show()
