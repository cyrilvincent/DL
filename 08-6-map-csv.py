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

tf.enable_eager_execution()

dataset = (tf.data.TextLineDataset(filename)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))  # Transform each elem by applying decode_csv fn

for row in dataset:
    print(row)



