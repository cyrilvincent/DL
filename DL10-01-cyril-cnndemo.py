import tensorflow.keras as keras

model = keras.models.load_model("data/dogsvscats/cyrilmodel.h5")
img = keras.preprocessing.image.load_img("data/dogsvscats/small/validation/cats/cat.1001.jpg", target_size=(224, 224))
img = keras.preprocessing.image.img_to_array(img)
img *= 1. / 255
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
res = model.predict(img)[0][0] # image 0, output 0
s = "Dog"
if res < 0.5:
    s = "Cat"
    res = 1 - res
print(f"Prediction: {s} ({res*100:.0f}%) with validation accuracy 90%")





