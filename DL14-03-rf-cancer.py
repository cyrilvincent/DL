import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import sklearn.model_selection as ms
import sklearn.ensemble as rf
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.neural_network as nn



# Charger breast-cancer/data.csv avec pandas read_csv(...,index_col="id")
# y = dataframe["diagnosis"] 0=benin 1 = malin
# x = dataframe.drop("diagnosis",1).drop("id")

# Sur x
# Chercher 2 colonnes qui gère la taille et la concavité
# Calculer sur ces 2 colonnes min,max,mean,std,median,quantile
# Retenez la valeurs mean et std
# Trouver une correlation
# x_benin = 0
# x_malin = 1

dataframe = pd.read_csv("data/breast-cancer/data.csv", index_col="id")
print(dataframe)
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

np.random.seed(0)
xtrain, xtest, ytrain, ytest = ms.train_test_split(x,y,train_size=0.8, test_size=0.2)

#model = svm.SVC(C=1, kernel="rbf")
model = rf.RandomForestClassifier()

model.fit(xtrain, ytrain)
plt.bar(x.columns, model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

print(model.score(xtest, ytest))

tree.export_graphviz(model.estimators_[0],
                out_file='data/breast-cancer/tree.dot',
                feature_names = x.columns,
                class_names = ["0", "1"],
                rounded = True, proportion = False,
                precision = 2, filled = True)


