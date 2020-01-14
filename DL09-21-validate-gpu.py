import tensorflow as tf
print(tf.config.experimental.list_physical_devices())
print(tf.config.experimental.list_physical_devices("GPU"))

# To disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# NVidia tool : nvidia-smi