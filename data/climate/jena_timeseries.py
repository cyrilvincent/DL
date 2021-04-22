import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

df = pd.read_csv("jena_climate_2009_2016.csv")
print(df)
# TS 10 min )> 1 hour
df = df[5::6]
df.to_csv("jena_climate_2009_2016_hourly.csv")
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(df.describe())