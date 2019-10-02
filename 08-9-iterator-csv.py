import tensorflow as tf

FEATURES = [
    'loyer',
]

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

filename = "house/house.csv"

def decode_csv(line):
    parsed_line = tf.decode_csv(line, [[0.], [0.]]) # Default values
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
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.Session()
for i in range(100):
  value = sess.run(next_element)
  print(value)



