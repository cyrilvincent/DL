import pandas as pd
import sklearn.linear_model as sklm

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

print(X.shape) #569 * 30
print(y.shape) #569

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

regr = sklm.LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_test, y_test))
print(regr.coef_)
print(regr.intercept_)
predict = regr.predict(X_test)







