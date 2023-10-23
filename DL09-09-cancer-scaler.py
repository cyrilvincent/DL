import tensorflow as tf
import pandas
import sklearn.preprocessing


tf.random.set_seed(1)

dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop("diagnosis", 1)

scaler = sklearn.preprocessing.RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])

model.compile(loss="mse",metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=20, batch_size=1, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)

