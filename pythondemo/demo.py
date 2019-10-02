print("Hello World!")

def add(i,j):
    return i + j

i = 1
print(type(i))

f = add

l = [1,2,3,4,9,8]
l = range(1000)

def isEven(x):
    return x % 2 == 0

import math
def isCosPositive(x):
    return math.cos(x) > 0

def xsinx(x):
    return x * math.sin(x)

def tanh(x):
    return math.tanh(x / 3)

# <=>

tanh = lambda x : math.tanh(x / 3)

def filterFn(l, fn):
    res = []
    for i in l:
        if fn(i):
            res.append(i)
    return res

def mapFn(l, fn):
    res = []
    for i in l:
        res.append(fn(i))
    return res

print(filterFn(l,isCosPositive))
print(mapFn(l, xsinx))
print(mapFn(l, lambda x : x * math.sin(x)))

import functools


print(type(f))

print(f(2,3))