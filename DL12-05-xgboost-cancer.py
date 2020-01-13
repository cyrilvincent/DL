import sklearn
import sklearn.datasets
cancer = sklearn.datasets.load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

print(X.shape) #569 * 30
print(y.shape) #569

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8, test_size=0.2)

import xgboost
model = xgboost.sklearn.XGBClassifier()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)

predicted = model.predict(X_test)
print(predicted)
print(y_test)
print(predicted - y_test)




