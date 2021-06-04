import sklearn.datasets
import sklearn.preprocessing
from sklearn.datasets import load_breast_cancer
import tensorflow.keras as keras
import tensorflow as tf
import pandas
import sklearn.preprocessing as pp

tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

scaler = sklearn.preprocessing.RobustScaler()
scaler.fit(x)
X = scaler.transform(x)



import tensorflow as tf
import tensorflow.keras as keras

y = keras.utils.to_categorical(y)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
  ])

model.compile(loss="categorical_crossentropy",metrics=['accuracy'])
# for 2 categories bce ~= cce
model.summary()

history = model.fit(X, y, epochs=100, batch_size=10, validation_split=0.2)
eval = model.evaluate(X, y)
print(eval)
print(f"Total accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

