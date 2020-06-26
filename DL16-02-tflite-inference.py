import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="data/h5/cholletmodel-mnist.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with np.load("data/mnist/mnist.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

# Test the model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)

# Random
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print(input_data.shape)

# Or the first image, be carefull dtype is mandatory because float64 instead
input_data = np.zeros(input_shape, dtype=np.float32)
x_test = x_test / 255.
for x in range(28):
    for y in range(28):
        input_data[0][x][y][0] = x_test[0][x][y]

# Or by reshape
input_data = x_test[0].astype(np.float32).reshape(1,28,28,1)


interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)