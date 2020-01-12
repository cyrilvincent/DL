from PIL import Image, ImageStat
import json

path = f"chest_xray"

def scan(path):
    with open(f"{path}/db.json", "r") as f:
        db = json.loads(f.read())
    for item in db["data"]:
        file = f"{item['path']}/{item['name']}"
        print(f"Open {file}")
        im = Image.open(file)
        stat = ImageStat.Stat(im)
        item["lum"] = int(stat.mean[0])
        item["std"] = int(stat.stddev[0])
        item["lummin"] = stat.extrema[0][0]
        item["lummax"] = stat.extrema[0][1]
        item["size"] = im.size
    db["lummin"] = min([item["lummin"] for item in db["data"]])
    db["lummax"] = max([item["lummax"] for item in db["data"]])
    db["lum"] = int(sum([item["lum"] for item in db["data"]]) / db["count"])
    db["std"] = int(sum([item["std"] for item in db["data"]]) / db["count"])
    return db

if __name__ == '__main__':
    print(f"Scan {path}/db.json")
    db = scan(path)
    print(f"Update db.json")
    with open(f"{path}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))






