import sklearn.datasets as ds
mnist = ds.fetch_mldata('MNIST original',data_home='./mnist/')

import numpy as np
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

# On redimensionne les données sous forme d'images
images = data.reshape((-1, 28, 28))

# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)

# On affiche les images avec la prédiction associée
import matplotlib.pyplot as plt
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Target: %i' % target[value])

plt.show()
