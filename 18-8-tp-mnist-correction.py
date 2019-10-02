from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original",data_home='./mnist/')
X = mnist.data / 255. # Transforme les 256 niveaux de gris en [0.0,1.0]
y = mnist.target
split_size = int(X.shape[0]*0.7)
X_train, X_test = X[:split_size], X[split_size:]
print(X_train.shape)
y_train, y_test = y[:split_size], y[split_size:]
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(500,500,500))
mlp.fit(X_train,y_train)
print(mlp.score(X_test, y_test))
# Le score n'est pas terrible, nous verrons pourquoi plus tard (MSE ??)


