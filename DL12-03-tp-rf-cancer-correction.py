import sklearn
import sklearn.datasets
cancer = sklearn.datasets.load_breast_cancer() # more info : https://goo.gl/U2Uwz2

#input
X=cancer['data']
y=cancer['target']

print(X.shape) #569 * 30
print(y.shape) #569

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8, test_size=0.2)

import sklearn.ensemble as rf
model = rf.RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(score)

predicted = model.predict(X_test)
print(predicted)
print(y_test)
print(predicted - y_test)

print(cancer.feature_names)
features_importances = model.feature_importances_
print(features_importances)
zip = list(zip(cancer.feature_names, features_importances))
zip = zip[features_importances.argsort()]
print(zip)

import matplotlib.pyplot as plt
plt.bar(range(len(features_importances)),features_importances)
plt.show()

plt.figure()
sklearn.tree.plot_tree(model.estimators_[0],
                feature_names = cancer.feature_names,
                class_names = cancer.target_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)
plt.show()


