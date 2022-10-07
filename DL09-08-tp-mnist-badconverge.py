import tensorflow.keras as keras
import tensorflow as tf

import numpy as np

#D'après DL03-02-npz-mnist.py
with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train'] #(60000,28,28) => (60000,784)
    x_test, y_test = f['x_test'], f['y_test'] #10000

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255 # x_train = (x_train - 127.5) / 127.5
x_test /= 255

x_train = x_train.reshape(-1,28*28) # 28*28 = 784
x_test = x_test.reshape(-1,28*28)

sample = np.random.randint(60000, size=1000)
x_train = x_train[sample]
y_train = y_train[sample]

#Topologie
#MLP
#TrainSet : 48000
#ValidationSet = 12000
#TestSet : 10000
#HiddenLayer : 4
#Input : 28x28 = 784
#Output : 1
#Topologie en V en =
#epochs=? batch_size=1 et 10
#Comparer accuracy et val_accuracy
#20%


model = None #TODO

predicted = None #TODO résulat de la prédiction sur le jeu de test

import matplotlib.pyplot as plt
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


