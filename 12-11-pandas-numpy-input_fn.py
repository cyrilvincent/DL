import pandas as pd
import tensorflow as tf
from sklearn import datasets
import itertools

print("Model")
data = pd.read_csv('house/house.csv')

FEATURES = [
    'loyer',
]

LABEL = 'surface'

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

input_fn = tf.estimator.inputs.numpy_input_fn(data, batch_size=10, num_epochs=None, shuffle=False)





