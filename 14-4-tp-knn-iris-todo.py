import sklearn.datasets as ds
iris = ds.fetch_mldata('iris',data_home='./mnist/')

print(len(iris.data))

import numpy as np
sample = np.random.randint(150, size=100)
data = iris.data[sample]
target = iris.target[sample]
data = data[:, :2] # Enlève les 2 1ere colonnes

# TODO

x = float(input("Saisir un float entre 0 et 1 pour le ratio des pétales: "))
y = float(input("Saisir un float entre 0 et 1 pour le ratio des tiges: "))
array = np.array([[x,y]])

# TODO
