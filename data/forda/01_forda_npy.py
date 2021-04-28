import numpy as np

def read_ucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


x_train, y_train = read_ucr("FordA_TRAIN.txt")
x_test, y_test = read_ucr("FordA_TEST.txt")

np.save("FordA", (x_train, y_train, x_test, y_test), allow_pickle=True)

