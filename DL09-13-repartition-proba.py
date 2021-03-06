import numpy as np

linsm = lambda x : x/sum(x)
geosm = lambda x : x**2/sum(x**2)
softmax = lambda x : np.exp(x)/sum(np.exp(x))

scores = np.array([0.2, 0.4, 0.8, 0.95, 0.05,0,0.01,0.02,0.03,0.04])
print("Scores", np.round(scores * 100).astype(int))
print(sum(scores))
probas = linsm(scores)
print("Linear", np.round(probas * 100).astype(int))
print(sum(probas))
probas = geosm(scores)
print("Geo", np.round(probas * 100).astype(int))
print(sum(probas))
probas = softmax(scores)
print("Softmax", np.round(probas * 100).astype(int))
print(sum(probas))