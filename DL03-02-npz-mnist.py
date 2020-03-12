import numpy as np

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

select = np.random.randint(x_train.shape[0], size=12)

import matplotlib.pyplot as plt
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(x_train[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Target: %i' % y_train[value])

plt.show()
