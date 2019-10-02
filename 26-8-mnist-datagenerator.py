from sklearn.datasets import fetch_mldata
import tensorflow as tf
import tensorflow.keras as keras

mnist = fetch_mldata("MNIST original")
X = mnist.data
y = mnist.target
split_size = int(X.shape[0]*0.7)
X_train, X_test = X[:split_size], X[split_size:]

# Les traitements d'images nécéssitent un reshape de (60000,784) en (60000,28,28,1) le 1 siginifiant le nombre de canaux de couleurs
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255

Y = keras.utils.to_categorical(y)
Y_train, Y_test = Y[:split_size], Y[split_size:]

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=tf.nn.relu),
    keras.layers.Conv2D(32, (3, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=tf.nn.relu),
    keras.layers.Conv2D(64, (3, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),

    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(500, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dense(10, activation=tf.nn.softmax),
  ])

sgd = keras.optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'],
              callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=2)]) # Se voit à partie de 5 epochs

gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
test_gen = keras.preprocessing.image.ImageDataGenerator()
train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)
model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=10,
                    validation_data=test_generator, validation_steps=10000//64)

print(model.summary())
# 99.55% d'accuracy !
