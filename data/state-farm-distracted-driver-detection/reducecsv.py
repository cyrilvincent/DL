with open("vgg16-bottleneck-train.small.csv", "w") as o:
    with open("vgg16-bottleneck-train.large.csv") as f:
        i = 0
        for row in f:
            if i % 100 == 0:
                o.write(row)
                print(row)
            i += 1
print(f"Reduce {i} to {i // 100}")


