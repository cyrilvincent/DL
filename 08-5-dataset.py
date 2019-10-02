#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
tf.disable_v2_behavior()

# Dataset
# 4 Tensors de dimension 0 (valeur unique)
dataset0 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4]))
print(dataset0.output_types)  # ==> "tf.float32"
print(dataset0.output_shapes)  # ==> "()"
for el in dataset0:
    print(el)

# 4 Tensors de dimension 1 (liste) avec 10 éléments
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"
for el in dataset1:
    print(el)

# il est plus facile de nommer les composants
dataset2 = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random.uniform([4]),
    "b": tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset2.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset2.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

