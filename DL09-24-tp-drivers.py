import pandas as pd

print("Load CSV")
df = pd.read_csv("data/dogsvscats/vgg16-bottleneck-train.small.csv")
print(df)
