import matplotlib.pyplot as plt
import math

def plotFn(inputFn, plotrange, step):
    x = [x * step for x in plotrange]
    y = [inputFn(x * step) for x in plotrange]
    plt.plot(x, y)
    plt.show()

sigmoidfn = lambda x : 1 / (1 + math.e ** -x)
xsinfn = lambda x : x * math.sin(x)

print(type(xsinfn))

plotFn(math.tanh, range(-1000,1000), 0.01)
