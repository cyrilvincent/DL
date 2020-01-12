from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

print(X.shape) #569 * 30
print(y.shape) #569

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

import sklearn.ensemble as rf
model = None #TODO
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)

# TODO Pickle save

predicted = None # TODO predict
print(predicted)
print(y_test)

# TODO print predicted - y_test

# TODO print important features

model = None

# TODO pickle load

# TODO fit again
model.n_estimators *= 2

# TODO Show graphviz





