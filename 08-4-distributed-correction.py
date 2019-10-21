#import tensorflow as tf
import tensorflow.compat.v1 as tf

# Start 08-4-cluster-1 and 08-4-cluster-2 before
a = tf.constant(1)
b = tf.constant(2)
with tf.device("/job:myjob/task:2"):
    c = a + b
with tf.device("/job:myjob/task:1/cpu:0"):
    d = c * b
sess = tf.Session("grpc://192.168.1.70:2222")
result = sess.run(d)
print(result)
