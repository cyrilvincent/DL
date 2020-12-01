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
model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

model.compile(loss="binary_crossentropy",metrics=['accuracy'])
# binary_crossentropy ~= mse
model.summary()

history = model.fit(X, y, epochs=10, batch_size=10, validation_split=0.2)
eval = model.evaluate(X, y)
print(eval)
predict = model.predict(X)
print(predict)

