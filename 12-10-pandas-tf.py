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

def input_fn(dataframe, num_epochs=None, n_batch=128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: dataframe[k].values for k in FEATURES}),
        y=pd.Series(dataframe[LABEL].values),
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle)





