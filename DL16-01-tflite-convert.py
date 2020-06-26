import tensorflow as tf

model = tf.keras.models.load_model("data/h5/cholletmodel-mnist.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("data/h5/cholletmodel-mnist.tflite", "wb").write(tflite_model)