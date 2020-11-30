import pandas
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

#dataframe = pandas.read_csv("data/house/house.csv")
# with sqlite3.connect("data/house/house.db3") as conn:
#     dataframe = pandas.read_sql('select loyer,surface from house', conn)
dataframe = pandas.read_excel("data/house/house.xlsx")
loyerparm2 = dataframe.loyer / dataframe.surface
print(np.mean(loyerparm2))
print(np.std(loyerparm2))
print(np.median(loyerparm2))
plt.scatter(dataframe.surface,dataframe.loyer)
plt.show()

#Reprise 15h15
#xlrd

