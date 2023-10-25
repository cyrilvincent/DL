import pandas as pd

df = pd.read_csv("data/dogsvscats/vgg16-bottleneck-train.small.csv")
print(df.head())
x = df.iloc[:,2:]
y = df.iloc[:,1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.8, test_size=0.2)

import sklearn.ensemble as rf
model = rf.RandomForestClassifier(n_estimators=100)
print("Fit")
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

filter_importances = []
i = 0
sum = 0
features = list(model.feature_importances_) # 7*7*512 = 25088
while i < len(features): # 25088
    sum += features[i] # First layer = 64 kernels
    i+=1
    if i % 392 == 0: #25088 / 64 = 392
        filter_importances.append(sum)
        sum = 0

print(filter_importances)
print(filter_importances.index(max(filter_importances)))

import tensorflow.keras as keras
model = keras.applications.VGG16()
model = keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()

img = keras.preprocessing.image.load_img("data/dogsvscats/small/validation/cats/cat.1000.jpg", target_size=(224, 224))
img = keras.preprocessing.image.img_to_array(img)
img *= 1. / 255
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
res = model.predict(img)

import matplotlib.pyplot as plt
ix = 1
for _ in range(8):
    for _ in range(8):
        ax = plt.subplot(8, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{filter_importances[ix - 1] * 100:.2f}%")
        plt.imshow(res[0, :, :, ix-1], cmap='gray')
        ix += 1
plt.show()

plt.imshow(res[0, :, :, filter_importances.index(max(filter_importances))], cmap='gray')
plt.show()