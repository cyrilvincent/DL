import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

FEATURES = [
    'loyer',
]

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

filename = "house/house.csv"

def decode_csv(line):
    parsed_line = tf.io.decode_csv(records=line, record_defaults=[[0.], [0.]]) # Default values
    label = parsed_line[1]  # Label = Target = Colonne 1
    del parsed_line[-1]
    features = parsed_line  # Tout sauf la dernière colonne
    d = dict(zip(FEATURES, features)), label
    return d

dataset = (tf.data.TextLineDataset(filename)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))
dataset = dataset.shuffle(buffer_size=256)
dataset = dataset.repeat(10)  # Repeats dataset this # times
dataset = dataset.batch(32)  # Batch size to use
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
next_element = iterator.get_next()
sess = tf.compat.v1.Session()
for i in range(100):
  value = sess.run(next_element)
  print(value)



