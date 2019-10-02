import tensorflow as tf
import tensorflow.keras as keras

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2
X=cancer['data']
y=cancer['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(X_train.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
sgd = keras.optimizers.SGD(nesterov=True)
model.compile(loss="mse", optimizer=sgd)
model.summary()

# use import tensorflow.keras as keras instead of import keras
estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
input_name = model.input_names[0]
train_input_fn = tf.estimator.inputs.numpy_input_fn({input_name:X_train}, X_train, batch_size=10, num_epochs=None, shuffle=False)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=100)
eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn, steps=10)
eval = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
print(eval)
