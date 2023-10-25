from flask import Flask, request, jsonify
import tensorflow as tf
from keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

app = Flask(__name__)
model: tf.keras.Sequential = tf.keras.models.load_model("data/h5/cancer-mlp.h5")
model_driver: tf.keras.Sequential = tf.keras.models.load_model("data/h5/drivers_10_mlp.h5")

@app.route("/")
def root():
    return jsonify(tf.__version__)

@app.route("/cancer", methods = ['POST'])
def cancer():
    features = request.json
    # Normaliser features
    res = model.predict(features)
    print(res)
    return jsonify(float(res[0][0]))

@app.route("/predict", methods = ['POST'])
def predict():
    data = request.files['image'].stream.read()
    print("save temp.png")
    with open("temp.png", "wb") as f:
        f.write(data)
    model = MobileNetV2()
    image = load_img('temp.png', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    print('%s (%.2f%%)' % (label[1], label[2] * 100))
    return jsonify([label[1], float(label[2])])



if __name__ == '__main__':
    app.run(host="0.0.0.0")

