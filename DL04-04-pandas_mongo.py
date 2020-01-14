import pandas as pd
import matplotlib.pyplot as plt
import pymongo

with pymongo.MongoClient('localhost', 27017) as client:
    db = client.test
    cursor = db.house.find()
    house_data = pd.DataFrame(list(cursor))

plt.plot(house_data['surface'], house_data['loyer'], 'ro', markersize=4)
plt.show()

