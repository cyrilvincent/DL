from sklearn.datasets import fetch_mldata
import tensorflow as tf
import tensorflow.keras as keras

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X = mnist.data / 255.
y = mnist.target
split_size = int(X.shape[0]*0.7)
X_train, X_test = X[:split_size], X[split_size:]
print(X_train.shape)
Y = keras.utils.to_categorical(y)
Y_train, Y_test = Y[:split_size], Y[split_size:]

model = keras.Sequential([
    keras.layers.Dense(784, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(10, activation=tf.nn.softmax),
  ])

model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
trained = model.fit(X_train, Y_train, epochs=2, batch_size=120, validation_data=(X_test, Y_test))

print(model.summary())
# Mauvais résultat à cause de mse

