import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

class Rectangle:

  def __init__(self, width, length):
    self.width = width
    self.length = length

  def area(self):
    return self.width * self.length

  def perimeter(self):
    return 2 * (self.width + self.length)

class Square(Rectangle):

  def __init__(self, side):
    super().__init__(side, side)


r1 = Rectangle(3,2)
print(r1.area())
s1 = Square(4)
print(s1.area())



latent_dim = 64
class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim):
    super().__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Flatten(), # tf.keras.layers.Reshape(28*28)
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(28*28, activation='sigmoid'),
      tf.keras.layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()


