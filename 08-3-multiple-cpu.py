import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device("cpu:0"):
    a = tf.constant(1)
    b = tf.constant(2)
    c = a+b
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
result = sess.run(c)
print(result)
