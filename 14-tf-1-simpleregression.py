import tensorflow as tf

print("Model")

FEATURES = [
    'loyer',
]

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

model = tf.estimator.LinearRegressor(feature_columns=feature_cols)

def decode_csv(line):
    parsed_line = tf.decode_csv(line, [[0.], [0.]]) # Default values
    label = parsed_line[1]  # Label = Target = Colonne 1
    #labels = parsed_line[1:] # si plusieurs
    del parsed_line[-1]
    features = parsed_line  # Tout sauf la dernière colonne
    d = dict(zip(FEATURES, features)), label
    return d

def input_fn(file_path, perform_shuffle=False, repeat_count=1, batch_size=32):
    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))  # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

filename = "house/house.csv"

print("Train")
# Utilisation d'une lambda pour passer des paramètres
model.train(input_fn=lambda:input_fn(filename))

print("Evaluate")
eval_result = model.evaluate(input_fn=lambda:input_fn(filename))
average_loss = eval_result["average_loss"]
rmse = average_loss ** 0.5 # Root Mean Square Error

print(eval_result)

print("Prediction")
# Pas de séparation Train et Test
predict_results = model.predict(input_fn=lambda:input_fn(filename))
for res in list(predict_results)[:10]:
    print(res)






