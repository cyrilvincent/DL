import matplotlib as mpl
import pandas as pd

# https://www.tensorflow.org/tutorials/structured_data/time_series

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv("jena_climate_2009_2016.csv")
print(df)

# TS 10 min )> 1 hour
df = df[5::6]

# Replace -9999 => 0
wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0
max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0
df.to_csv("jena_climate_2009_2016_00.csv", index=False)
