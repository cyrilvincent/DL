import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

with sqlite3.connect("house/house.db3") as conn:
    house_data = pd.read_sql('select loyer,surface from house', conn)

# On affiche le nuage de points dont on dispose
plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()

