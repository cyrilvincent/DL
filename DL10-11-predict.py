import tensorflow.keras as keras
import os

# load the model
model = keras.models.load_model("data/dogsvscats/cyrilmodel.h5")

def predict(path):
    image = keras.preprocessing.image.load_img(path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image *= 1. / 255
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return model.predict(image)

def predicts(path):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    qs = []
    print(f"Scan {path}/1")
    files = os.listdir(f"{path}/1")
    for f in files :
        res = predict(f"{path}/1/{f}")[0][0]
        if res > 0.5 :
            tp += 1
            qs.append((res - 0.5) * 2)
        else :
            fp += 1
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
    print(f"Scan {path}/0")
    files = os.listdir(f"{path}/0")
    for f in files:
        res = predict(f"{path}/0/{f}")[0][0]
        # print(res)
        if res < 0.5:
            tn += 1
            qs.append((0.5 - res) * 2)
        else:
            fn += 1
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f }")
    return tp,tn,fp,fn,qs





if __name__ == '__main__':
    #res = predict("res/1.jpg")
    # res = predict("data/RevetementVetuste/1/A07079111281875000001_001_201808.jpg")
    # print(res)
    tp,tn,fp,fn,qs = predicts("data/RevetementVetuste")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
    print(f"Precision: {(tp) / (tp + fp):.2f}")
    print(f"Recall: {(tp) / (tp + fn):.2f}")
    print(sum(qs)/len(qs), min(qs), max(qs))
