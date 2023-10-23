import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.describe())
plt.scatter(dataframe.surface, dataframe.loyer)
plt.show()

# y = 1 colonne = diagnosis 1 = cancer
# x = tout sauf les colonnes diagnosis et id
# x = dataframe.drop("diagnosis", axis=1)
# dataframe[dataframe.diagnosis == 1].describe()


