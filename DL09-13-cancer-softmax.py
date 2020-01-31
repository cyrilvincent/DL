from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)



import tensorflow as tf
import tensorflow.keras as keras

y = keras.utils.to_categorical(y, 2)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
  ])

model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=['accuracy'])
# for 2 categories bce ~= cce
model.summary()

history = model.fit(X, y, epochs=200, batch_size=10, validation_split=0.2)
eval = model.evaluate(X, y)
print(eval)

