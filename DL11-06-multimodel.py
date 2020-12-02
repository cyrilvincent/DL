import tensorflow.keras as keras
import importlib
import time

class ImageCategorize:

    module_MobileNetV2 = importlib.import_module("tensorflow.keras.applications.mobilenet_v2")
    module_ResNet152V2 = importlib.import_module("tensorflow.keras.applications.resnet_v2")
    module_VGG16 = importlib.import_module("tensorflow.keras.applications.vgg16")
    module_Xception = importlib.import_module("tensorflow.keras.applications.xception")
    module_InceptionV3 = importlib.import_module("tensorflow.keras.applications.inception_v3")
    module_InceptionResNetV2 = importlib.import_module("tensorflow.keras.applications.inception_resnet_v2")
    module_NASNetLarge = importlib.import_module("tensorflow.keras.applications.nasnet")
    module_NASNetMobile = module_NASNetLarge

    def __init__(self):
        self.module = None

    def predict(self, path:str, model:str="MobileNetV2"):
        print(f"Predict with model {model}")
        self.module = eval(f"ImageCategorize.module_{model}")
        size = 224
        if "ception" in model:
            size = 299
        elif model == "NASNetLarge":
            size = 331
        model = eval(f"self.module.{model}()")
        image = keras.preprocessing.image.load_img(path, target_size=(size, size))
        image = keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = self.module.preprocess_input(image)
        predicts = model.predict(image)
        return predicts

    def labels(self, predicts):
        return self.module.decode_predictions(predicts)[0]

print("NP Image Categorization")
print("=======================")
cat = ImageCategorize()
img = "data/dogsvscats/small/validation/cats/cat.1000.jpg"
predicts = cat.predict(img)

t = time.perf_counter()
predicts = cat.predict(img) #2s, 20%
labels = cat.labels(predicts)
print(labels)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")

t = time.perf_counter()
predicts = cat.predict(img,"ResNet152V2") #7s, 73%
labels = cat.labels(predicts)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")

t = time.perf_counter()
predicts = cat.predict(img,"VGG16") #2s, 77%
labels = cat.labels(predicts)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")

t = time.perf_counter()
predicts = cat.predict(img,"Xception") #2s, 32%
labels = cat.labels(predicts)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")

t = time.perf_counter()
predicts = cat.predict(img,"InceptionV3") #4s, 27%
labels = cat.labels(predicts)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")

t = time.perf_counter()
predicts = cat.predict(img,"InceptionResNetV2") #9s, 53%
labels = cat.labels(predicts)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")

t = time.perf_counter()
predicts = cat.predict(img,"NASNetLarge") #15s, 78%
labels = cat.labels(predicts)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")

t = time.perf_counter()
predicts = cat.predict(img,"NASNetMobile") #10s, 22%
labels = cat.labels(predicts)
print(f"{labels[0][1]} {labels[0][2]*100:.1f}%")
print(f"Predict in {time.perf_counter() - t:.1f} s")
