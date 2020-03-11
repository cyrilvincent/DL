import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Reshape the dataset into 4D array
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

import sklearn.svm as svm
model = svm.SVC(C=0.1, kernel= "linear")
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('Score: %f' % score)

predicted = model.predict(x_test)

images = x_test.reshape((-1, 28, 28))

import numpy as np
select = np.random.randint(images.shape[0], size=12)

import matplotlib.pyplot as plt
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % predicted[value])

plt.show()

misclass = (y_test != predicted)
misclass_images = images[misclass,:,:]
misclass_predicted = predicted[misclass]

select = np.random.randint(misclass_images.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % misclass_predicted[value])

plt.show()
