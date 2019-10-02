import sklearn.datasets as ds
iris = ds.fetch_mldata('iris',data_home='./mnist/')

print(len(iris.data))

import numpy as np
sample = np.random.randint(150, size=100)
data = iris.data[sample]
target = iris.target[sample]
data = data[:, :2]

import sklearn.model_selection as ms
xtrain, xtest, ytrain, ytest = ms.train_test_split(data, target, train_size=0.8, test_size=0.2)

import sklearn.ensemble as rf
model = rf.RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)
error = 1 - model.score(xtest, ytest)
print('Erreur: %f' % error)


features_importances = model.feature_importances_
print(features_importances)

import matplotlib.pyplot as plt

# On récupère les prédictions sur les données test
predicted = model.predict(xtest)

x = float(input("Saisir un float entre 0 et 1 pour le ratio des pétales: "))
y = float(input("Saisir un float entre 0 et 1 pour le ratio des tiges: "))
x = [[x,y]]
array = np.array(x)
print(model.predict(array))
