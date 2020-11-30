print("Hello World!")

i = 3
print(type(i))

if i % 2 == 0:
    print("pair")
else:
    print("impair")

mylist = [1,5,6,7,9,10,12]
for val in mylist:
    print(val)

print(mylist[2:5])

# Afficher les nombre pairs d'une liste
mylist = [1,5,6,7,9,10,12]
res = []
for val in mylist:
    if val % 2 == 0:
        res.append(val)
print(res)