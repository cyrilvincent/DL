import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

print("Hello World")
print(pd.__version__)
print(np.__version__)
print(sklearn.__version__)

dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.describe())


x = dataframe.surface.values.reshape(-1,1)
y = dataframe.loyer

model = lm.LinearRegression()
model.fit(x, y)
predicted = model.predict(x)
print(model.coef_, model.intercept_)

plt.scatter(x, y)
plt.plot(x, predicted, color="red")
plt.show()

