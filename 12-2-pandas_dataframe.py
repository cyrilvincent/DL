import numpy as np
import pandas as pd

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)

# A partir d'une Series
df = pd.DataFrame(population, columns=['population'])
print(df)

# A partir d'une list de dict
data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
df = pd.DataFrame(data)
print(df)

# A partir d'un tableau Numpy de dimension 2
df = pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])
print(df)

# Une fonction pour générer facilement des DataFrame.
# Elle nous sera utile dans la suite de ce chapitre.
def make_df(cols, ind):
    """Crée rapidement des DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)

# exemple
make_df('ABC', range(3))