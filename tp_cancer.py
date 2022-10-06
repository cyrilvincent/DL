import pandas as pd
import sklearn.neighbors as nn

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe.describe().T)
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

# Créer le modèle
# Fitter le modéle
# Prédire x
# Afficher le score
# IL existe des biais importants (lié au dataset)
# Il existe un risque important lié au predict et au score

