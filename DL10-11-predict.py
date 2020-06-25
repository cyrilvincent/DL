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
    print(f"Scan {path}/dogs")
    files = os.listdir(f"{path}/dogs")
    for f in files :
        res = predict(f"{path}/dogs/{f}")[0][0]
        if res > 0.5 :
            tp += 1
            qs.append((res - 0.5) * 2)
        else :
            fp += 1
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
    print(f"Scan {path}/cats")
    files = os.listdir(f"{path}/cats")
    for f in files:
        res = predict(f"{path}/cats/{f}")[0][0]
        # print(res)
        if res < 0.5:
            tn += 1
            qs.append((0.5 - res) * 2)
        else:
            fn += 1
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
    return tp,tn,fp,fn,qs





if __name__ == '__main__':
    tp,tn,fp,fn,qs = predicts("data/dogsvscats/small/train")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.2f}")
    print(f"Precision: {(tp) / (tp + fp):.2f}")
    print(f"Recall: {(tp) / (tp + fn):.2f}")
    print("Proba mean, min, max", sum(qs)/len(qs), min(qs), max(qs))
