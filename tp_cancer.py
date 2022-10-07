import pickle

import pandas as pd
import sklearn.neighbors as nn
import sklearn.ensemble as rf
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import numpy as np
import sklearn.svm as svm

np.random.seed(0)


dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe.describe().T)
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y, train_size=0.8, test_size=0.2)

# model = nn.KNeighborsClassifier(n_neighbors=3)
# model = rf.RandomForestClassifier(max_depth=5, n_estimators=10)
model = svm.SVC(C=1, kernel="sigmoid", degree=1)
model.fit(xtrain, ytrain)
predicted = model.predict(xtest)
print(predicted)
print(model.score(xtrain, ytrain))
print(model.score(xtest, ytest))

with open("data/breast-cancer/model.pickle", "wb") as f:
    pickle.dump(model, f)

model = None

with open("data/breast-cancer/model.pickle", "rb") as f:
    model = pickle.load(f)

# Créer le modèle
# Fitter le modéle
# Prédire x
# Afficher le score
# IL existe des biais importants (lié au dataset)
# Il existe un risque important lié au predict et au score

# print(model.feature_importances_)
#
# plt.bar(x.columns, model.feature_importances_)
# plt.xticks(rotation=45)
# plt.show()



# import sklearn.tree as tree
# tree.export_graphviz(model.estimators_[0], out_file="data/breast-cancer/tree.dot", feature_names=x.columns,
#                      class_names=["0", "1"], rounded=True, filled=True)

