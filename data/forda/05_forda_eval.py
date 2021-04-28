import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

x_train, y_train, x_test, y_test = np.load("FordA_norm.npy", allow_pickle=True)
model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)