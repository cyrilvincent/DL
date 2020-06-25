import pandas as pd

print("Load CSV")
df = pd.read_csv("data/state-farm-distracted-driver-detection/vgg16-bottleneck-train.small.csv")
print(df)
