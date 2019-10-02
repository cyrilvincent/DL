# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# On charge le dataset
house_data = pd.read_csv('house/house.csv')

surface_max = 300
standard_dev = 2199

f = lambda x : 41 * x - 283

print(house_data)

# Filtrer le nuage de point par surface_max et 3 * standard_dev

# On affiche le nuage de points dont on dispose
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()

