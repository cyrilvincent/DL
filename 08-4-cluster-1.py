import tensorflow as tf
cluster = tf.train.ClusterSpec({"worker": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="worker", task_index=0)
print("OK")
server.join()
# rpc://job:myjob/task:0

