import numpy as np
X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
y=np.array([0,1,1,0],dtype=float)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(3,3,3), activation="tanh", learning_rate="adaptive", alpha=1e-5)
mlp.fit(X,y)

print(mlp.score(X, y))
predictions = mlp.predict(X)
print(predictions)