#import sklearn.datasets as ds
#iris = ds.fetch_mldata('iris',data_home='./mnist/')
from sklearn.datasets import load_iris
iris = load_iris()

print(len(iris.data))

import numpy as np
sample = np.random.randint(150, size=100)
data = iris.data[sample]
target = iris.target[sample]
#data = data[:, :2]

import sklearn.model_selection as ms
xtrain, xtest, ytrain, ytest = ms.train_test_split(data, target, train_size=0.8, test_size=0.2)

import sklearn.ensemble as rf
model = rf.RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)
print('Erreur: %f' % model.score(xtest, ytest))

estimator = model.estimators_[0]

import sklearn.tree
# Export as dot file
sklearn.tree.export_graphviz(estimator, out_file='tree.dot',
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)

# Il faut alors visualiser dans graphviz
