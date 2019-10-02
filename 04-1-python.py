import csv

data = []
with open("house.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = (float(row["loyer"]), float(row["surface"]))
        data.append(t)

print(data)

loyers_par_m2 = [d[0]/d[1] for d in data]
print(loyers_par_m2)
moyenne = sum(loyers_par_m2)/len(loyers_par_m2)
print(moyenne)

import matplotlib.pyplot as plt

surfaces = [d[1] for d in data]
loyers = [d[0] for d in data]



plt.plot(surfaces, loyers, 'ro', markersize=4)
plt.show()

# regression = ax + b
# a vers 25 +/- 50
# b vers 0 +/- 1000
arange = range(20,50)
brange = range(-400,400, 2)

import math

regressionab = lambda a,b,x:a*x+b

theta = (0,0)
var_min = math.pow(2,31)
error_rate_min = 0
for a in arange:
    print(str((a-arange[0])*100/len(arange))+"%")
    for b in brange:
        i = 0
        errors = []
        for x in surfaces:
            loyer_theorique = regressionab(a,b,x)
            loyer_reel = loyers[i]
            i+=1
            error_square = (loyer_reel - loyer_theorique) ** 2
            error_rate = math.sqrt(error_square) / loyer_reel
            errors.append(error_square)
        var = sum(errors)/len(errors)
        if var < var_min:
            var_min = var
            error_rate_min = error_rate
            theta = (a,b)

print(theta)
print(math.sqrt(var_min))
print(str(error_rate_min*100)+"%")

plt.plot(surfaces, loyers, 'ro', markersize=4)
plt.plot(range(400), [41*x-286 for x in range(400)] )
plt.show()


# Exclusion des cas en dehors de 3ET




