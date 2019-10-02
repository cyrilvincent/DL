#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib
tf.disable_v2_behavior()
print(device_lib.list_local_devices())
print(tf.config.experimental_list_devices())

with tf.device("/CPU:0"):
    a = tf.constant(1)
    b = tf.constant(2)
    c = a+b
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
result = sess.run(c)
print(result)
