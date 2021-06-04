import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf

BATCH_SIZE = 64 #64
EPOCH_SIZE = 64
NB_EPOCH = 5 #5
SIZE = 128

tf.random.set_seed(1)
np.random.seed(1)


# https://colab.research.google.com/drive/169pJ-xECBWDW9Q92naaNE3oRyBr7D-uh#scrollTo=Q_xGx5xngv_r
# https://medium.com/analytics-vidhya/object-localization-using-keras-d78d6810d0be

def circle_gen(batch_size=BATCH_SIZE):
    # enable generating infinite amount of batches
    while True:
        # generate black images in the wanted size
        X = np.zeros((batch_size, SIZE, SIZE, 3))
        Y = np.zeros((batch_size, 3))
        # fill each image
        for i in range(batch_size):
            x = np.random.randint(8, SIZE - 8)
            y = np.random.randint(8, SIZE - 8)
            a = min(SIZE - max(x, y), min(x, y))
            r = np.random.randint(4, a)
            for x_i in range(SIZE):
                for y_i in range(SIZE):
                    if ((x_i - x) ** 2) + ((y_i - y) ** 2) < r ** 2:
                        X[i, x_i, y_i, :] = 1
            Y[i, 0] = (x - r) / SIZE
            Y[i, 1] = (y - r) / SIZE
            Y[i, 2] = 2 * r / SIZE
        yield X, Y


vgg = keras.applications.VGG16(input_shape=[SIZE, SIZE, 3], include_top=False, weights='imagenet')
# for layer in vgg.layers:
#     layer.trainable = False
x = keras.layers.Flatten()(vgg.output)
x = keras.layers.Dense(3, activation='sigmoid')(x)
model = keras.models.Model(vgg.input, x)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001))
# What about the loss function? the output of a sigmoid can be treated as probabilistic values, and therefore we can use binary_crossentropy loss
# Usually, this loss is being used with binary values as the ground-truth ({0,1}), but it doesn’t have to be this way- we can use values from [0,1]. For our use, the ground-truth values are indeed in the range [0,1], since it represents location inside an image and dimensions
model.fit(circle_gen(), steps_per_epoch=EPOCH_SIZE, epochs=NB_EPOCH)

# given image and a label, plots the image + rectangle
def plot_pred(img, p):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = Rectangle(xy=(p[1] * SIZE, p[0] * SIZE), width=p[2] * SIZE, height=p[2] * SIZE, linewidth=1, edgecolor='g',
                     facecolor='none')
    ax.add_patch(rect)
    plt.show()


# generate new image
x, _ = next(circle_gen())
# predict
pred = model.predict(x)
# examine 1 image
im = x[0]
p = pred[0]
plot_pred(im, p)
