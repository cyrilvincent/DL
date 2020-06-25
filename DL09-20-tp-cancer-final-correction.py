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
    keras.layers.Dropout(0.1), #Peu utile car trop peu de données
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

#SGD ne marche pas car pas assez de données
model.compile(loss="binary_crossentropy", metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=100, batch_size=100, validation_split=0.2)
print(model.evaluate(X, y))
model.save("data/h5/cancer.h5")

