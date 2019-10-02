import pandas as pd
import sklearn.linear_model as sklm

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

print(cancer.feature_names)
print(X.shape) #569 * 30
print(y.shape) #569
print(y)

