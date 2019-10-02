import numpy as np

def f(x):
    """ function to approximate by polynomial interpolation"""
    return  2.5 * x * np.sin(x * 0.7) + 2

# generate points used to plot
x_plot = np.linspace(0, 10, 100)
# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)
# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

import matplotlib.pyplot as plt
plt.scatter(x_plot, f(x_plot))
plt.show()

f = lambda x, a, b, c: a * x * np.sin(x * b) + c

import scipy.optimize as opt
print(x.shape)
print(y.shape)
weigths, conv = opt.curve_fit(f, x, y)
print(weigths)
print(conv)



