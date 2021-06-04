import pandas as pd

print("Load CSV")
df = pd.read_csv("data/state-farm-distracted-driver-detection/vgg16-bottleneck-train.small.csv")
print(df)
x = df.iloc[:,2:]
y = df.iloc[:,1]
x=x[y < 2]
y=y[y < 2]
