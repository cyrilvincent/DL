import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(np.__version__)
print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe_filtered = dataframe[dataframe.surface < 170]
print(dataframe_filtered)
print(dataframe_filtered.describe())
loyer_per_m2 = dataframe_filtered.loyer / dataframe_filtered.surface
print(loyer_per_m2.describe())

plt.scatter(dataframe_filtered.surface, dataframe_filtered.loyer)
plt.show()

