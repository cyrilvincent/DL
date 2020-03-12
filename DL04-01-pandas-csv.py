# On importe les librairies dont on aura besoin pour ce tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# On charge le dataset
house_data = pd.read_csv('data/house/house.csv')

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()