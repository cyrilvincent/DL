import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('house/house.csv')
regr = sklm.LinearRegression()
X = data["surface"].values.reshape(-1,1)
y = data["loyer"]

import sklearn.model_selection as ms

xtrain, xtest, ytrain, ytest = ms.train_test_split(X, y, train_size=0.8, test_size=0.2)

import sklearn.linear_model as sklm

regr = sklm.LinearRegression()
regr.fit(xtrain,ytrain)
import math
print(regr.score(xtest,ytest))
print(regr.coef_)
print(regr.intercept_)

import matplotlib.pyplot as plt
plt.plot(xtrain, ytrain, 'ro', markersize=4)
plt.plot(xtrain, regr.predict(xtrain) )
plt.show()

plt.plot(xtest, ytest, 'ro', markersize=4)
plt.plot(xtest, regr.predict(xtest) )
plt.show()

print(regr.score(xtrain,ytrain))

import sklearn.metrics as m
print(math.sqrt(m.mean_squared_error(regr.predict(xtest), ytest)))





