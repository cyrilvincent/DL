#import tensorflow as tf
import tensorflow.compat.v1 as tf
cluster = tf.train.ClusterSpec({"myjob": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="myjob", task_index=1)
print("OK")
server.join()
# /job:local/myjob:1