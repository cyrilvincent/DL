import csv

data = []
with open("house.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = (float(row["loyer"]), float(row["surface"]))
        data.append(t)

print(data)

# Filtre
import math
data = [d for d in data if d[1]<=300 and math.fabs(41*d[1]-286 - d[0])<3*911]

import matplotlib.pyplot as plt

surfaces = [d[1] for d in data]
loyers = [d[0] for d in data]


arange = range(30,50)
brange = range(-350,-200)

regressionab = lambda a,b,x:a*x+b

theta = (0,0)
et_min = math.pow(2,31)
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
            error = (loyer_reel - loyer_theorique)**2
            error_rate = error / loyer_reel
            errors.append(error)
        et = sum(errors)/len(errors)
        if et < et_min:
            et_min = et
            error_rate_min = error_rate
            theta = (a,b)

print(theta)
print(math.sqrt(et_min))
print(str(error_rate_min*100)+"%")

plt.plot(surfaces, loyers, 'ro', markersize=4)
plt.plot(range(250), [28*x+299 for x in range(250)] )
plt.show()


# Exclusion des cas en dehors de 3ET




