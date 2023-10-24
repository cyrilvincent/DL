import tensorflow as tf
import pandas


dataframe = pandas.read_csv("data/breast-cancer/data.csv", index_col="id")
y = dataframe.diagnosis
x = dataframe.drop(["diagnosis"], 1)

ko = dataframe[dataframe.diagnosis == 1]
ok = dataframe[dataframe.diagnosis == 0]

print(ok.describe().T)
print(ko.describe().T)

