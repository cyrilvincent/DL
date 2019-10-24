import tensorflow as tf
task_index = 0 #TODO
cluster = tf.train.ClusterSpec({"worker": ["192.168.1.00:2222", #TODO
                                          "192.168.1.00:2222",
                                          "192.168.1.00:2222",
                                          "192.168.1.00:2222",
                                          "192.168.1.00:2222",
                                          "192.168.1.00:2222",
                                          "192.168.1.00:2222",
                                          ]})
server = tf.train.Server(cluster, job_name="worker", task_index=task_index)
server.join()
# /job:myjob/task:0
# Start on each PC