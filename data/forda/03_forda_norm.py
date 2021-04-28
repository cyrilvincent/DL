import numpy as np

x_train, y_train, x_test, y_test = np.load("FordA.npy", allow_pickle=True)

num_classes = len(np.unique(y_train))

# label = -1, 1 => 0,1
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# Univariate => Multivariate
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Shuffle
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

np.save("FordA_norm", (x_train, y_train, x_test, y_test), allow_pickle=True)


