import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 #opencv-python

def add_square(img, center, width):
  start_point = (center[0]-width, center[1]-width)
  end_point   = (center[0]+width, center[1]+width)
  cv2.rectangle(img, start_point, end_point, random_color(),cv2.FILLED)

def create_data_sample(num_shapes, height, width):
  img = np.zeros((height, width, 3))
  label = np.zeros((height, width, 1))
  for _ in range(num_shapes):
    x = int(np.random.rand()*width)
    y = int(np.random.rand()*height)
    if np.random.rand() > 0.5:
      cv2.circle(img, (x,y), int(np.ceil(width*0.05)), random_color(), cv2.FILLED)
      cv2.circle(label, (x,y), int(np.ceil(width*0.02)), (1,1,1), cv2.FILLED)
    else:
      add_square(img, (x,y), int(np.ceil(width*0.05*0.7)))
  return img, label


def test_batch(num_shapes, height, width):
    img, label = create_data_sample(num_shapes, height, width)
    return np.expand_dims(img, 0), np.expand_dims(label, 0)


def grayscale_image(img):
    return np.expand_dims((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3, 2)


# (128,128,1) --> (128,128,3)
def to_three_channels(img):
    return np.squeeze(np.stack((img, img, img), 2))


def random_color():
    return np.random.rand(3)


def show_sample(img, label):
    figure, axis = plt.subplots(1, 2)
    axis[0].imshow(img)
    axis[1].imshow(to_three_channels(label))


def show_batch(img, label):
    show_sample(img[0, :, :, :], label[0, :, :, :])


img_shape = (128, 128, 3)
img, label = create_data_sample(10, img_shape[0], img_shape[1])
show_sample(img, label)
plt.show()


inputs = keras.Input(shape=img_shape)
conv1 = keras.layers.Conv2D(16,(5,5),padding='same',activation='relu')(inputs)
conv1 = keras.layers.BatchNormalization(momentum=0.99)(conv1)
conv2 = keras.layers.Conv2D(32,(5,5),padding='same',activation='relu')(conv1)
conv2 = keras.layers.BatchNormalization(momentum=0.99)(conv2)
outputs = keras.layers.Conv2D(1,(5,5),padding='same',activation='relu')(conv2)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

def loss_function(y_true, y_pred):
  squared_diff = tf.square(y_true - y_pred)
  return tf.reduce_mean(squared_diff)

x_list = []
y_list = []
for i in range(1000):
  x,y  = create_data_sample(10, img_shape[0], img_shape[1])
  x_list.append(x)
  y_list.append(y)
x_list = np.array(x_list)
y_list = np.array(y_list)

opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss=loss_function)
model.fit(x_list, y_list, batch_size=100, epochs=100)

img, label = test_batch(10, img_shape[0], img_shape[1])
y_pred = model.predict(img)
print(y_pred.max())
show_batch(img, y_pred)
plt.show()
