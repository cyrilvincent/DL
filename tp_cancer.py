import pandas as pd
import sklearn.neighbors as nn
import sklearn.ensemble as rf
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe.describe().T)
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

# model = nn.KNeighborsClassifier(n_neighbors=3)
model = rf.RandomForestClassifier()
model.fit(x, y)
predicted = model.predict(x)
print(predicted)
print(model.score(x, y))
# Créer le modèle
# Fitter le modéle
# Prédire x
# Afficher le score
# IL existe des biais importants (lié au dataset)
# Il existe un risque important lié au predict et au score

print(model.feature_importances_)

plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()



import sklearn.tree as tree
tree.export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns,
                     class_names=["0", "1"], rounded=True, filled=True)

