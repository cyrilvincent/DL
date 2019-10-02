import pandas as pd
import sklearn.linear_model as sklm

data = pd.read_csv('house/house.csv')
regr = sklm.LinearRegression()
X = data["surface"].values.reshape(-1,1)
y = data["loyer"]
regr.fit(X,y)
print(regr.predict(X))
print(regr.score(X, y))
print(regr.coef_)
print(regr.intercept_)

import matplotlib.pyplot as plt
plt.plot(data["surface"], data["loyer"], 'ro', markersize=4)
plt.plot(data["surface"], regr.predict(X) )
plt.show()






