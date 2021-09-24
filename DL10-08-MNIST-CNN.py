import tensorflow.keras as keras

import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), input_shape=(28, 28, 1), padding="same")) # 28,28,16
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) # 14,14,16

model.add(keras.layers.Conv2D(16, (3, 3))) # 10,10,16
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) # 5,5,16

#Dense
model.add(keras.layers.Flatten()) # 400
model.add(keras.layers.Dense(128))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

hist = model.fit(x=x_train,y=y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))

#model.save_weights('data/h5/cholletmodel-mnist-weights.h5') # 97.5%
model.save('data/h5/cholletmodel-mnist.h5') # 97.5%

import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.plot([None] + hist.history['accuracy'], 'o-')
ax.plot([None] + hist.history['val_accuracy'], 'x-')
ax.legend(['Train accuracy', 'Validation accuracy'], loc = 0)
ax.set_title('Training/Validation accuracy per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.show()

test_score = model.evaluate(x_test, y_test)
print("Test accuracy {:.2f}%".format(test_score[1] * 100))
