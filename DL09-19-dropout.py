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

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

sgd = keras.optimizers.SGD(nesterov=True, lr=1e-4)
model.compile(loss="binary_crossentropy", optimizer=sgd,metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=200, batch_size=10, validation_split=0.2)
eval = model.evaluate(X, y)
print(eval)

model.save("data/h5/cancer-mlp.h5")
model.save_weights("data/h5/cancer-mlp-weights.h5")

model.load_weights("data/h5/cancer-mlp-weights.h5") #Work only with the same model
model = keras.models.load_model("data/h5/cancer-mlp.h5") #No need to have de model structure

