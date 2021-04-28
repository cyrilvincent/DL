import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing
import scipy
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_cm(y_true, y_pred, class_names):
  cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots(figsize=(18, 16))
  ax = sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap=sns.diverging_palette(220, 20, n=7),
      ax=ax
  )

  plt.ylabel('Actual')
  plt.xlabel('Predicted')
  ax.set_xticklabels(class_names)
  ax.set_yticklabels(class_names)
  b, t = plt.ylim() # discover the values for bottom and top
  b += 0.5 # Add 0.5 to the bottom
  t -= 0.5 # Subtract 0.5 from the top
  plt.ylim(b, t) # update the ylim(bottom, top) values
  plt.show() # ta-da!

model = keras.models.load_model("cnn.h5")
score_cnn = model.evaluate(X_test.reshape(-1, X_train.shape[1], X_train.shape[2], 1), y_test)
print(f"Score CNN: {score_cnn}")
y_pred = model.predict(X_test.reshape(-1, X_train.shape[1], X_train.shape[2], 1))

plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)

model = keras.models.load_model("lstm.h5")
score_cnn = model.evaluate(X_test, y_test)
print(f"Score LSTM: {score_cnn}")
y_pred = model.predict(X_test)

plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)

