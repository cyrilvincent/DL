import sklearn.datasets
import sklearn.preprocessing

cancer = sklearn.datasets.load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']

scaler = sklearn.preprocessing.RobustScaler()
scaler.fit(X)
X = scaler.transform(X)



import tensorflow as tf
import tensorflow.keras as keras

model = keras.models.load_model("data/h5/cancer.h5")

print(model.evaluate(X, y))
predicted = model.predict(X)

predicted = predicted.reshape(-1)
print(predicted)
errors = predicted - y
print(errors)
print(len(errors[errors == 0]) / len(y))

import sklearn.metrics
print(sklearn.metrics.confusion_matrix(y,predicted))
print(sklearn.metrics.classification_report(y,predicted))
