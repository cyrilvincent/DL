import pandas as pd
import tensorflow as tf
from sklearn import datasets
import itertools

print("Model")
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X=cancer['data']
y=cancer['target']

print(cancer.feature_names)
print(X.shape) #569 * 30
print(y.shape) #569

feature_cols = [tf.feature_column.numeric_column(k) for k in cancer.feature_names]

def input_fn(dataframe, num_epochs=None, n_batch=128, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=X,
        y=y,
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle)

model = tf.estimator.LinearRegressor(feature_columns=feature_cols)

print("Train")
model.train(input_fn=input_fn(cancer,1,128,False))

print("Evaluate")
eval_result = model.evaluate(input_fn=input_fn(cancer,1,128,False))
average_loss = eval_result["average_loss"]
rmse = average_loss ** 0.5 # Root Mean Square Error

print(eval_result)

print("Prediction")
# Pas de séparation Train et Test
predict_results = model.predict(input_fn=input_fn(cancer))
