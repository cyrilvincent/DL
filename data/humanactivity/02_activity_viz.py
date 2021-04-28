import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

df = pd.read_csv('WISDM_clean.csv')
df_train = df[df['user_id'] <= 30]
df_test = df[df['user_id'] > 30]

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

sns.countplot(x = 'activity',
              data = df,
              order = df.activity.value_counts().index)
plt.show()


sns.countplot(x = 'user_id',
              data = df,
              palette=[sns.color_palette()[0]],
              order = df.user_id.value_counts().index);
plt.title("Records per user");
plt.show()


def plot_activity(activity, df):
  data = df[df['activity'] == activity][['x_axis', 'y_axis', 'z_axis']][:200]
  axis = data.plot(subplots=True, figsize=(16, 12),
                   title=activity)
  for ax in axis:
    ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
  plt.show()

plot_activity("Sitting", df)
plot_activity("Jogging", df);

