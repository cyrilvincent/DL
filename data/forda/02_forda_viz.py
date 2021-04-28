import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_train, y_train, x_test, y_test = np.load("FordA.npy", allow_pickle=True)

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()

pos = x_train[y_train == 1][0]
neg = x_train[y_train == -1][0]
fft = tf.signal.rfft(pos)
f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(pos)
f = f_per_dataset/n_samples_h
plt.plot(np.abs(fft))
plt.show()
fft = tf.signal.rfft(neg)
f_per_dataset = np.arange(0, len(fft))
n_samples_h = len(neg)
f = f_per_dataset/n_samples_h
plt.plot(np.abs(fft))
plt.show()


