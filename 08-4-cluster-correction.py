import tensorflow as tf
task_index = 0 #TODO
cluster = tf.train.ClusterSpec({"worker": ["192.168.1.76:2222",
                                          "192.168.1.66:2222",
                                          "192.168.1.107:2222",
                                          "192.168.1.70:2222",
                                          "192.168.1.112:2222",
                                          "192.168.1.57:2222",
                                          "192.168.1.86:2222",
                                          ]})
server = tf.train.Server(cluster, job_name="worker", task_index=task_index)
server.join()
# /job:myjob/task:0
# Start on each PC