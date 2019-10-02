def filterFn(l, fn):
    res = []
    for i in l:
        if fn(i):
            res.append(i)
    return res

def mapFn(l, fn):
    for i in l:
        yield fn(i)

def list(l):
    res = []
    for i in l:
        res.append(i)
    return res

import math
length = 1000
l = range(length)
res = mapFn(l, lambda x : math.tanh(x / length))
res2 = mapFn(res, lambda x : max(x , 0))
res3 = mapFn(res2, lambda x : 1 / (1 + math.e ** -x))



print("result")
for i in res:
    print(i)


