import tensorflow.keras as keras
import tensorflow as tf

import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# Set numeric type to float32 from uint8
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

y_train = keras.utils.to_categorical(y_train) #[0,1,2,3,4,5,6,7,8,9]
y_test = keras.utils.to_categorical(y_test)

sample = np.random.randint(60000, size=5000)
x_train = x_train[sample]
y_train = y_train[sample]

model = None #TODO
# 10 outputs + sigmoide dernier layer + categorical_crossentropy
# input 2 => [0,0,1,0,0,0,0,0,0,0]
# output = [0,0,1,0,0,0,0,0,0,0]

predicted = model.predict(x_test)
import matplotlib.pyplot as plt
# Gestion des erreurs
# on récupère les données mal prédites
misclass = (y_test != predicted.reshape(-1))
images = x_test.reshape((-1, 28, 28))
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

