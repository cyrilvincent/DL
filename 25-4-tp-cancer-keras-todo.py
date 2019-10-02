import tensorflow as tf
import tensorflow.keras as keras

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test)

#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
model = None #TODO

model.compile() # TODO
model.summary()

print(X_train.shape)
history = None # TODO

print(history)

print(X_test.shape)
print(y_test.shape)
eval = model.evaluate(X_test, y_test)
print(eval)
