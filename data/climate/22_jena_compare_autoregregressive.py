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

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)

  def warmup(self, inputs):
      # inputs.shape => (batch, time, features)
      # x.shape => (batch, lstm_units)
      x, *state = self.lstm_rnn(inputs)

      # predictions.shape => (batch, features)
      prediction = self.dense(x)
      return prediction, state

  def call(self, inputs, training=None):
      # Use a TensorArray to capture dynamically unrolled outputs.
      predictions = []
      # Initialize the lstm state
      prediction, state = self.warmup(inputs)

      # Insert the first prediction
      predictions.append(prediction)

      # Run the rest of the prediction steps
      for n in range(1, self.out_steps):
          # Use the last prediction as input.
          x = prediction
          # Execute one lstm step.
          x, state = self.lstm_cell(x, states=state,
                                    training=training)
          # Convert the lstm output to a prediction.
          prediction = self.dense(x)
          # Add the prediction to the output
          predictions.append(prediction)

      # predictions.shape => (time, batch, features)
      predictions = tf.stack(predictions)
      # predictions.shape => (batch, time, features)
      predictions = tf.transpose(predictions, [1, 0, 2])
      return predictions

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
prediction, state = feedback_model.warmup(multi_window.example[0])
print(prediction.shape)
print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)


history = wg.compile_and_fit(feedback_model, multi_window)

multi_window.plot(feedback_model)
plt.show()

