import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import data.climate.window_generator as wg

# https://www.tensorflow.org/tutorials/structured_data/time_series

train_df = pd.read_csv("jena_climate_2009_2016_train.csv")
val_df = pd.read_csv("jena_climate_2009_2016_val.csv")
test_df = pd.read_csv("jena_climate_2009_2016_test.csv")

w2 = wg.WindowGenerator(input_width=6, label_width=1, shift=1, train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=['T (degC)'])
print(w2)

example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')
# Le code ci-dessus a pris un lot de 3 fenêtres à 7 pas de temps, avec 19 fonctionnalités à chaque pas de temps. Il les a divisés en un lot de 6 pas de temps, 19 entrées de fonction et une étiquette à 1 étape de fonction. L'étiquette n'a qu'une seule fonctionnalité car le WindowGenerator été initialisé avec label_columns=['T (degC)']

w2.example = example_inputs, example_labels
w2.plot()
plt.show()

w2.plot(plot_col='p (mbar)')
plt.show()
