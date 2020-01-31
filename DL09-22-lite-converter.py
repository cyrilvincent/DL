import tensorflow.lite as lite

# Convert HDF5 to tflite
converter = lite.TFLiteConverter.from_keras_model_file("keras_model.h5")
tflite_model = converter.convert()
with open("h5/converted_model.tflite", "wb") as f:
    f.write(tflite_model)
