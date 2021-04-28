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

OUT_STEPS = 24
multi_window = wg.WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
                               input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])


history = wg.compile_and_fit(multi_dense_model, multi_window)

multi_dense_model.save("h5/dense_32_24_19__32_24_19.h5")

multi_window.plot(multi_dense_model)
plt.show()

