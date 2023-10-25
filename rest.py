from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model: tf.keras.Sequential = tf.keras.models.load_model("data/h5/cancer-mlp.h5")
model_driver: tf.keras.Sequential = tf.keras.models.load_model("data/h5/drivers_10_mlp.h5")

@app.route("/")
def root():
    return jsonify(tf.__version__)

@app.route("/cancer", methods = ['POST'])
def cancer():
    features = request.json
    res = model.predict(features)
    print(res)
    return jsonify(float(res[0][0]))

@app.route("/driver", methods = ['POST'])
def driver():
    data = request.files['image'].stream.read()
    with open("temp.png", "wb") as f:
        f.write(data)


if __name__ == '__main__':
    app.run(host="0.0.0.0")

