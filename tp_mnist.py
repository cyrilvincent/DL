import numpy as np
import sklearn.neighbors as nn

np.random.seed(0)

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

sample = np.random.randint(60000, size=5000)
x_train = x_train[sample]
y_train = y_train[sample]

model = nn.KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))

predicted = model.predict(x_test)

images = x_test.reshape((-1, 28, 28))

select = np.random.randint(images.shape[0], size=12)

import matplotlib.pyplot as plt
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % predicted[value])

plt.show()

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