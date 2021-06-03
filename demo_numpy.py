import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# pip install scikit-learn

print(np.__version__)
print(pd.__version__)

dataframe = pd.read_csv("data/house/house.csv")
dataframe_filtered = dataframe[dataframe.surface < 170]
print(dataframe_filtered)
print(dataframe_filtered.describe())
loyer_per_m2 = dataframe_filtered.loyer / dataframe_filtered.surface
print(loyer_per_m2.describe())

# dataframe_filtered.surface.shape = (545)
x = dataframe_filtered.surface.values.reshape(-1, 1) # (545,1)
y = dataframe_filtered.loyer

# instanciation
model = lm.LinearRegression() # f(x) = ax + b (a et b sont random)
# Fit
model.fit(x, y) # f(x) = ax + b  avec loss (convergergence)
# Predict, score = accuracy
predict = model.predict(x)
print("Accuracy", model.score(x, y))
# Proprietaire Linear
print(model.coef_, model.intercept_)
f = lambda x: x + 1 # <=> f(x) = x + 1
f = lambda x: model.coef_ * x + model.intercept_
plt.scatter(dataframe_filtered.surface, dataframe_filtered.loyer)
plt.plot(dataframe_filtered.surface, predict, color='red')
plt.plot(dataframe_filtered.surface, f(dataframe_filtered.surface), color='green')
plt.show()

