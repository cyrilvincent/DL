import tensorflow as tf
cluster = tf.train.ClusterSpec({"worker": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="worker", task_index=1)
print("OK")
server.join()
# /job:local/myjob:1