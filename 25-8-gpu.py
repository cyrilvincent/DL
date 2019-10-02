import tensorflow as tf
import keras

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
print(K.tensorflow_backend._LOCAL_DEVICES)

# Force la prise en compte de 2 GPU et 4 CPU
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Il suffit ensuite de créer le model (ici sans GPU et un seul CPU !)
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 1} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)
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

history = model.fit(X_train, y_train, epochs=500)
eval = model.evaluate(X_test, y_test)
print(eval)
