import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

sample = np.random.randint(60000, size=500)
x_train = x_train[sample]
y_train = y_train[sample]

x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

import sklearn.neighbors as nn
model = nn.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('Score: %f' % score)

# On récupère les prédictions sur les données test
predicted = model.predict(x_test)

# On redimensionne les données sous forme d'images
images = x_test.reshape((-1, 28, 28))

import matplotlib.pyplot as plt

# Gestion des erreurs
# on récupère les données mal prédites
misclass = (y_test != predicted)
misclass_images = images[misclass,:,:]
misclass_predicted = predicted[misclass]

# on sélectionne un échantillon de ces images
select = np.random.randint(misclass_images.shape[0], size=12)

# on affiche les images et les prédictions (erronées) associées à ces images
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % misclass_predicted[value])

plt.show()
