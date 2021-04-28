import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pandas as pd
import data.climate.window_generator as wg
import numpy as np

# https://www.tensorflow.org/tutorials/structured_data/time_series

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

train_df = pd.read_csv("jena_climate_2009_2016_train.csv")
val_df = pd.read_csv("jena_climate_2009_2016_val.csv")
test_df = pd.read_csv("jena_climate_2009_2016_test.csv")

val_performance = {}
performance = {}
single_step_window = wg.WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=1, label_width=1, shift=1,
    label_columns=['T (degC)'])

CONV_WIDTH = 3
conv_window = wg.WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T (degC)'])

wide_window = wg.WindowGenerator(train_df=train_df, val_df=val_df, test_df=test_df,
    input_width=24, label_width=24, shift=1,
    label_columns=['T (degC)'])

dense = keras.models.load_model("h5/dense_32_1_19__32_1_19.h5")
val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

lstm_model = keras.models.load_model("h5/lstm_32_24_19__32_24_19.h5")
val_performance['LSTM'] = lstm_model.evaluate(conv_window.val)
performance['LSTM'] = lstm_model.evaluate(conv_window.test, verbose=0)

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()