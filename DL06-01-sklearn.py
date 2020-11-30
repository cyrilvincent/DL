import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('data/house/house.csv')
model = sklm.LinearRegression() #f(x)=ax+b
X = data.surface.values.reshape(-1,1) #545 => (545,1)
y = data.loyer
model.fit(X, y)
print(model.predict(X))
print(model.score(X, y))
print(model.coef_)
print(model.intercept_)

import matplotlib.pyplot as plt
plt.plot(data.surface, data.loyer, 'ro', markersize=4)
plt.plot(data.surface, model.predict(X))
plt.show()






