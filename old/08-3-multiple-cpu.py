import tensorflow as tf
print(tf.config.experimental.list_physical_devices())

tf.debugging.set_log_device_placement(True)
with tf.device("/device:CPU:0"):
    a = tf.constant(1)
    b = tf.constant(2)
    c = a+b
print(c)