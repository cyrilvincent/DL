import numpy as np

def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)

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

import sklearn.linear_model as sklm
model = sklm.LinearRegression()
model.fit(X,y)
y_plot = model.predict(X_plot)
plt.scatter(x_plot, f(x_plot))
plt.plot(x_plot, y_plot)
plt.show()
import math
print(model.score(X, y)) #75%

import sklearn.preprocessing as pp
import sklearn.pipeline as pipe

# make_pipe_line create a model with a feature
# PolynomialFeature = Algo polynomial
# Ridge = Error square algo
# model = pp.make_pipeline(pipe.PolynomialFeatures(1), sklm.Ridge())
# <=> LinearRegression
model = pipe.make_pipeline(pp.PolynomialFeatures(2), sklm.Ridge())
model.fit(X,y)
y_plot = model.predict(X_plot)
plt.scatter(x_plot, f(x_plot))
plt.plot(x_plot, y_plot)
plt.show()
import math
print(model.score(X, y)) #72%

model = pipe.make_pipeline(pp.PolynomialFeatures(3), sklm.Ridge())
model.fit(X,y)
y_plot = model.predict(X_plot)
plt.scatter(x_plot, f(x_plot))
plt.plot(x_plot, y_plot)
plt.show()
import math
print(model.score(X, y)) #61%

model = pipe.make_pipeline(pp.PolynomialFeatures(4), sklm.Ridge())
model.fit(X,y)
y_plot = model.predict(X_plot)
plt.scatter(x_plot, f(x_plot))
plt.plot(x_plot, y_plot)
plt.show()
import math
print(model.score(X, y)) #18%

model = pipe.make_pipeline(pp.PolynomialFeatures(5), sklm.Ridge())
model.fit(X,y)
y_plot = model.predict(X_plot)
plt.scatter(x_plot, f(x_plot))
plt.plot(x_plot, y_plot)
plt.show()
import math
print(model.score(X, y)) #3%

model = pipe.make_pipeline(pp.PolynomialFeatures(6), sklm.Ridge())
model.fit(X,y)
y_plot = model.predict(X_plot)
plt.scatter(x_plot, f(x_plot))
plt.plot(x_plot, y_plot)
plt.show()
import math
print(model.score(X, y)) #3%

# Boucle
for i in range(10):
    model = pipe.make_pipeline(pp.PolynomialFeatures(i), sklm.Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.scatter(x_plot, f(x_plot))
    plt.plot(x_plot, y_plot, label="d%d" % i)
plt.legend(loc='lower left')
plt.show()

model = pipe.make_pipeline(pp.FunctionTransformer(np.sin), sklm.Ridge())
model.fit(X,y)
y_plot = model.predict(X_plot)
plt.scatter(x_plot, f(x_plot))
plt.plot(x_plot, y_plot)
plt.show()
import math
print(model.score(X, y)) #3%
