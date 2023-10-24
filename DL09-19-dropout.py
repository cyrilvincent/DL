import sklearn.preprocessing
import tensorflow as tf
import pandas


tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

scaler = sklearn.preprocessing.RobustScaler()
scaler.fit(x)
X = scaler.transform(x)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

sgd = tf.keras.optimizers.SGD(nesterov=True, lr=1e-4)
model.compile(loss="binary_crossentropy",metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=20, batch_size=10, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)

model.save("data/h5/cancer-mlp.h5")

#model = tf.keras.models.load_model("data/h5/cancer-mlp.h5") #No need to have de model structure

