f = lambda x : 3.5*x + 4
import numpy as np
import matplotlib.pyplot as plt

l = np.array([f(x) for x in range(0,1000,10)]) + np.random.normal(0,100,100)

import scipy.stats as st
t = st.linregress(range(0,1000,10), l)
print(t)

plt.scatter(range(0,1000,10), l)
plt.plot(range(1000), [t.slope*x+t.intercept for x in range(1000)] )
plt.show()