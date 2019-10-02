from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original',data_home='./mnist/')

print(mnist.data.shape) # 70000 éléments de 784 points (28*28)
print(mnist.target.shape) #70000 targets (labels)

import numpy as np
# sampling
l = np.arange(10) * 2 # 0,2,3,6,....,20
s = np.array([0,1,5,8])
print(l[s]) # 0,2,10,16

sample = np.random.randint(70000, size=5000) # 5000 éléments compris en 0 et 70000
data = mnist.data[sample]
target = mnist.target[sample]

# On redimensionne les données sous forme d'images
images = data.reshape((-1, 28, 28))

# On selectionne un echantillon de 24 images au hasard
select = np.random.randint(images.shape[0], size=24)

# On affiche les images avec la prédiction associée
import matplotlib.pyplot as plt
for index, value in enumerate(select):
    plt.subplot(4,6,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title(f'Target: {int(target[value])}')
plt.show()

