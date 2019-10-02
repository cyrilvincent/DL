from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.neural_network import MLPClassifier
mlp = None #TODO
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

#print(predictions)
print(mlp.score(X_test, y_test))

print(mlp.coefs_)
