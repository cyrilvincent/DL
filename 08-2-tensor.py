import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
c = a+b
print(c.numpy())

v1 = tf.constant([1,2])
v2 = tf.constant([3,4])
print(v1.shape)
v3 = v1 * v2
print(v3.numpy())

c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)
print(e.numpy())

@tf.function
def f(b):
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

b = tf.Variable(12.)
f(b) #Implicitly call to numpy()