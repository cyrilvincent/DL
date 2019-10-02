
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[500, 500, 500],
    optimizer=tf.train.AdamOptimizer(1e-5),
    n_classes=10,
)

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.train)[0]},
    y=input(mnist.train)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn, steps=1000)
eval = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
print(classifier.evaluate(input_fn=test_input_fn))

