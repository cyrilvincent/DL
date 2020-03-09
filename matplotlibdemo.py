import matplotlib.pyplot as plt
import numpy as np
v1 = np.arange(1000)

plt.plot(v1,np.sin(v1 / 100))
plt.show()