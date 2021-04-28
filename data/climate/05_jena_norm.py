import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# https://www.tensorflow.org/tutorials/structured_data/time_series

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv("jena_climate_2009_2016_02.csv")
train_df = pd.read_csv("jena_climate_2009_2016_train.csv")
val_df = pd.read_csv("jena_climate_2009_2016_val.csv")
test_df = pd.read_csv("jena_climate_2009_2016_test.csv")

# Normalisation
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Verif de la normalisation
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.show()

train_df.to_csv("jena_climate_2009_2016_train.csv", index=False)
val_df.to_csv("jena_climate_2009_2016_val.csv", index=False)
test_df.to_csv("jena_climate_2009_2016_test.csv", index=False)
