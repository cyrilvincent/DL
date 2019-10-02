import sklearn.datasets as ds
iris = ds.fetch_mldata('iris',data_home='./mnist/')

print(len(iris.data))

import numpy as np
sample = np.random.randint(150, size=100)
data = iris.data[sample]
target = iris.target[sample]
data = data[:, :2] # Enlève les 2 1ere colonnes
print(data)

import sklearn.model_selection as ms
xtrain, xtest, ytrain, ytest = ms.train_test_split(data, target, train_size=0.8, test_size=0.2)

import sklearn.neighbors as nn
model = nn.KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print('Score: %f' % score)

scores = []
for k in range(2,15):
    model = nn.KNeighborsClassifier(k)
    scores.append(model.fit(xtrain, ytrain).score(xtest, ytest))
import matplotlib.pyplot as plt
plt.plot(range(2,15), scores, 'o-')
plt.show()

min_nn = scores.index(max(scores)) + 2
print("min_nn: "+str(min_nn))

model = nn.KNeighborsClassifier(n_neighbors=min_nn)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print('Score: %f' % score)

# On récupère le classifieur le plus performant
model = nn.KNeighborsClassifier(min_nn)
model.fit(xtrain, ytrain)

# On récupère les prédictions sur les données test
predicted = model.predict(xtest)

print(predicted)
print(xtest)
x = float(input("Saisir un float entre 0 et 1 pour le ratio des pétales: "))
y = float(input("Saisir un float entre 0 et 1 pour le ratio des tiges: "))
x = [[x,y]]
array = np.array(x)
print(model.predict(array))
