import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import scipy
import tensorflow.keras as keras

df = pd.read_csv('WISDM_clean.csv')
df_train = df[df['user_id'] <= 30]
df_test = df[df['user_id'] > 30]

# Norm
scale_columns = ['x_axis', 'y_axis', 'z_axis']

scaler = sklearn.preprocessing.RobustScaler()

scaler = scaler.fit(df_train[scale_columns])

df_train.loc[:, scale_columns] = scaler.transform(
  df_train[scale_columns].to_numpy()
)

df_test.loc[:, scale_columns] = scaler.transform(
  df_test[scale_columns].to_numpy()
)


def create_dataset(X, y, time_steps=1, step=1):
  Xs, ys = [], []
  for i in range(0, len(X) - time_steps, step):
    v = X.iloc[i:(i + time_steps)].values
    labels = y.iloc[i: i + time_steps]
    Xs.append(v)
    ys.append(scipy.stats.mode(labels)[0][0])
  return np.array(Xs), np.array(ys).reshape(-1, 1)


TIME_STEPS = 200
STEP = 40

X_train, y_train = create_dataset(
    df_train[['x_axis', 'y_axis', 'z_axis']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[['x_axis', 'y_axis', 'z_axis']],
    df_test.activity,
    TIME_STEPS,
    STEP
)

print(X_train.shape, y_train.shape)

enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)

enc = enc.fit(y_train)

y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(8, (2, 2), input_shape=(200, 3, 1)))
# 200, 3, 8

#Dense
model.add(keras.layers.Flatten())
# 200*3*8

model.add(keras.layers.Dense(units=512, activation='relu'))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(units=128, activation='relu'))
model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(
    X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1), y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    shuffle=True
)
model.summary()
model.save("cnn.h5")
