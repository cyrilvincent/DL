import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import data.climate.window_generator as wg

# https://www.tensorflow.org/tutorials/structured_data/time_series

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

train_df = pd.read_csv("jena_climate_2009_2016_train.csv")
val_df = pd.read_csv("jena_climate_2009_2016_val.csv")
test_df = pd.read_csv("jena_climate_2009_2016_test.csv")

num_features = train_df.shape[1]

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features),
])

single_step_window = wg.WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=1, label_width=1, shift=1)

history = wg.compile_and_fit(dense, single_step_window)

dense.save("h5/dense_32_1_19__32_1_19.h5")

wide_window = wg.WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=24, label_width=24, shift=1)
wide_window.plot(dense, plot_col="p (mbar)")
wide_window.plot(dense, plot_col="T (degC)")
plt.show()

