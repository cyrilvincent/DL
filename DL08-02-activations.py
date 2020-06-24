import math

constantFn = lambda x : 1
identityFn = lambda x: x
reluFn = lambda x: max(0, x)
hardSigmoidFn = lambda x: min(max(0, x + 2), 4)
leluFn = lambda x, alpha: x * alpha if x < 0 else x
tanhFn = lambda x: 0.5 + 0.5 * math.tanh(x / 2)
sigmoidFn = lambda x: 1 / (1 + math.exp(x * -1))


import matplotlib.pyplot as plt
l = [x / 100 for x in range(-1000,1000)]
plt.subplot(231)
plt.plot(l, [identityFn(x) for x in l])
plt.title("Identity")
plt.subplot(232)
plt.plot(l, [reluFn(x) for x in l])
plt.title("Relu")
plt.subplot(233)
plt.plot(l, [hardSigmoidFn(x) for x in l])
plt.title("Hard sigmoïd")
plt.subplot(234)
plt.plot(l, [leluFn(x, 0.1) for x in l])
plt.title("Leaky relu")
plt.subplot(235)
plt.plot(l, [tanhFn(x) for x in l])
plt.title("Tanh")
plt.subplot(236)
plt.plot(l, [sigmoidFn(x) for x in l])
plt.title("Sigmoïd")
plt.show()




