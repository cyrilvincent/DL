import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 #opencv-python

num_shapes = 2
img_shape = (128, 128, 3)

def create_data_sample(num_shapes, height, width):
    img = np.zeros((height, width, 3))
    labels = []
    for _ in range(num_shapes):
        x = int(np.random.rand()*width)
        y = int(np.random.rand()*height)
        cv2.circle(img, (x,y), int(np.ceil(width*0.05)), np.random.rand(3), cv2.FILLED)
        labels.append(x)
        labels.append(y)
    return img, labels

img, labels = create_data_sample(num_shapes, img_shape[0], img_shape[1])
plt.imshow(img)
labels = np.array(labels)
plt.scatter(labels[::2], labels[1::2], color="red")
plt.show()


model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

for layer in model.layers:
    layer.trainable = False

x = model.output
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(30, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(num_shapes * 2)(x)
model.summary()

x_list = []
y_list = []
for i in range(10000):
    x, y = create_data_sample(num_shapes, img_shape[0], img_shape[1])
    x_list.append(x)
    y_list.append(y)
x_list = np.array(x_list)
y_list = np.array(y_list, dtype=np.float32)

def loss_function(y_true, y_pred):
  squared_diff = tf.square(y_true - y_pred)
  return tf.reduce_mean(squared_diff)

model.compile(loss=loss_function)
model.fit(x_list, y_list, batch_size=100, epochs=100)
# loss = 43.3 = 6.5px d'erreur

y_pred = model.predict(x_list)
print(y_pred)
plt.imshow(x_list[0])
plt.scatter(y_pred[0, ::2], y_pred[0, 1::2], color="red")
plt.show()
