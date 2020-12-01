import tensorflow.keras as keras
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = keras.models.load_model("data/dogsvscats/cholletmodel.h5")
model = keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()

img = keras.preprocessing.image.load_img("data/dogsvscats/small/validation/cats/cat.1000.jpg", target_size=(150, 150))
img = keras.preprocessing.image.img_to_array(img)
img *= 1. / 255
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
res = model.predict(img)

ix = 1
for _ in range(4):
	for _ in range(8):
		ax = plt.subplot(4, 8, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(res[0, :, :, ix-1], cmap='gray')
		ix += 1
plt.show()

model = keras.models.load_model("data/dogsvscats/cholletmodel.h5")
model = keras.models.Model(inputs=model.inputs, outputs=model.layers[4].output)
model.summary()

img = keras.preprocessing.image.load_img("data/dogsvscats/small/validation/cats/cat.1000.jpg", target_size=(150, 150))
img = keras.preprocessing.image.img_to_array(img)
img *= 1. / 255
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
res = model.predict(img)

ix = 1
for _ in range(4):
	for _ in range(8):
		ax = plt.subplot(4, 8, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(res[0, :, :, ix-1], cmap='gray')
		ix += 1
plt.show()

model = keras.models.load_model("data/dogsvscats/cholletmodel.h5")
model = keras.models.Model(inputs=model.inputs, outputs=model.layers[8].output)
model.summary()

img = keras.preprocessing.image.load_img("data/dogsvscats/small/validation/cats/cat.1000.jpg", target_size=(150, 150))
img = keras.preprocessing.image.img_to_array(img)
img *= 1. / 255
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
res = model.predict(img)

ix = 1
for _ in range(8):
	for _ in range(8):
		ax = plt.subplot(8, 8, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(res[0, :, :, ix-1], cmap='gray')
		ix += 1
plt.show()


