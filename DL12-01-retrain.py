import pandas as pd

df = pd.read_csv("data/dogsvscats/vgg16-bottleneck-train.large.csv") # Train with new data
print(df.head())
x = df.iloc[:,2:]
y = df.iloc[:,1]

#Fix the random seed
#Do it when you generate h5
import numpy as np
np.random.seed(1511)
import tensorflow
tensorflow.random.set_seed(1511)

import tensorflow.keras as keras
model = keras.models.load_model("data/dogsvscats/cyrilmodel.h5")
model.summary()

history = model.fit(x, y, epochs=200, batch_size=10, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)

