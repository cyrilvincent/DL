import pandas as pd
import data.climate.window_generator as wg

# https://www.tensorflow.org/tutorials/structured_data/time_series

train_df = pd.read_csv("jena_climate_2009_2016_train.csv")
val_df = pd.read_csv("jena_climate_2009_2016_val.csv")
test_df = pd.read_csv("jena_climate_2009_2016_test.csv")

# Fenetrage
# Regroupement des données avec un pas
w1 = wg.WindowGenerator(input_width=24, label_width=1, shift=24, train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=['T (degC)'],)
print(w1)
w2 = wg.WindowGenerator(input_width=6, label_width=1, shift=1, train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=['T (degC)'])
print(w2)

