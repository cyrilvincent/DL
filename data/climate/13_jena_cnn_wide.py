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

CONV_WIDTH = 3
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = wg.WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['T (degC)'])


conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

history = wg.compile_and_fit(conv_model, wide_conv_window)

conv_model.save("h5/cnn_32_26_19__32_24_1.h5")

wide_conv_window.plot(conv_model)
plt.show()

