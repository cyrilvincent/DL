import pandas as pd

df = pd.read_csv("data/dogsvscats/vgg16-bottleneck-train.small.csv")
print(df.head())
x = df.iloc[:,2:]
y = df.iloc[:,1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.8, test_size=0.2)

import sklearn.svm as svm
model = svm.SVC(C=0.1, kernel= "linear")
print("Fit")
model.fit(X_train, y_train)

print("Score")
score = model.score(X_test, y_test)
print(score)
