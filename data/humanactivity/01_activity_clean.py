import numpy as np
import pandas as pd

column_names = [
  'user_id',
  'activity',
  'timestamp',
  'x_axis',
  'y_axis',
  'z_axis'
]

# https://www.cis.fordham.edu/wisdm/includes/datasets/latest/

df = pd.read_csv(
  'WISDM_ar_v1.1_raw.txt',
  header=None,
  names=column_names
)

df.z_axis.replace(regex=True, inplace=True, to_replace=r';', value=r'')
df['z_axis'] = df.z_axis.astype(np.float64)
df.dropna(axis=0, how='any', inplace=True)
print(df)
df.to_csv("WISDM_clean.csv")