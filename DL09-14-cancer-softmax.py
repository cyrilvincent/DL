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

y = tf.keras.utils.to_categorical(y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(30, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
  ])

model.compile(loss="categorical_crossentropy",metrics=['accuracy'])
# for 2 categories bce ~= cce
model.summary()

history = model.fit(x, y, epochs=20, batch_size=1, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)
print(f"Total accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

