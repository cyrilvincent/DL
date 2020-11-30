import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-2*np.pi,2*np.pi,0.01)
y = np.sin(x)
y2 = np.cos(x)
y3 = np.tanh(x)

plt.plot(x,y) #scatter, bar
plt.plot(x,y2) #scatter, bar
plt.plot(x,y3) #scatter, bar
plt.show()

