import tensorflow as tf
tf.compat.v1.enable_eager_execution()

filenames = ["house/house.csv"]
record_defaults = [tf.float32] * 2   # Two required float columns
record_defaults = [[0.0]] * 2 # Two optional columns with 0 as default value
dataset = tf.data.experimental.CsvDataset(filenames, record_defaults, header=True)
print(dataset)
for el in dataset:
  print(el)

# Cycle indéfiniment sur le dataset
dataset = tf.data.experimental.make_csv_dataset(filenames, batch_size = 1, header=True, column_defaults = record_defaults, )
print(dataset)
for el in dataset:
  print(el)