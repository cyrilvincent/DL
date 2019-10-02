import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
c = a+b
sess = tf.Session()
result = sess.run(c)
print(result)



v1 = tf.constant([1,2])
v2 = tf.constant([3,4])
print(v1.shape)
v3 = v1 * v2
sess = tf.Session()
result = sess.run(v3)
print(result)

c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)
sess = tf.Session()
result = sess.run(e)
print(result)
