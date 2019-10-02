# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# On charge le dataset
house_data = pd.read_csv('house/house.csv')

surface_max = 300
house_data = house_data[house_data.surface < surface_max]
standard_dev = 2199
f = lambda x : 41 * x - 283
house_data = house_data[abs(f(house_data.surface) - house_data.loyer) < 3 * standard_dev ]
print(house_data)

import scipy.stats as stats
theta = stats.linregress(house_data.surface, house_data.loyer)
print(theta)

# On affiche le nuage de points dont on dispose
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.plot(range(surface_max), [theta.slope * x + theta.intercept for x in range(surface_max)])
plt.show()

