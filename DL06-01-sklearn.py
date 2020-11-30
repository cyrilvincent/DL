import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('data/house/house.csv')
model = sklm.LinearRegression() #f(x)=ax+b
X = data.surface[data.surface < 200].values.reshape(-1,1) #545 => (545,1)
y = data.loyer[data.surface < 200]
model.fit(X, y)
print(model.predict(X))
print(model.score(X, y))
print(model.coef_)
print(model.intercept_)

import matplotlib.pyplot as plt
plt.plot(data.surface[data.surface<200], data.loyer[data.surface<200], 'ro', markersize=4)
plt.plot(data.surface[data.surface<200], model.predict(X))
plt.show()






