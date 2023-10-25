import tensorflow as tf

model = tf.keras.models.load_model("data/h5/cholletmodel-mnist.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # 8 bits par défaut (integer)
# converter.target_spec.supported_types = [tf.float16] # Passage en 16 bits
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE] # Convertit les poids ET les fonctions d'activation en 8 bits
tflite_model = converter.convert()
open("data/h5/cholletmodel-mnist-q16.tflite", "wb").write(tflite_model)