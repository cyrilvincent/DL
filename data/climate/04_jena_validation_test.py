import matplotlib as mpl
import pandas as pd

# https://www.tensorflow.org/tutorials/structured_data/time_series

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv("jena_climate_2009_2016_02.csv")

# Test + Validation
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

train_df.to_csv("jena_climate_2009_2016_train.csv", index=False)
val_df.to_csv("jena_climate_2009_2016_val.csv", index=False)
test_df.to_csv("jena_climate_2009_2016_test.csv", index=False)

