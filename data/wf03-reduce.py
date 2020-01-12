import json
import numpy as np
from PIL import Image, ImageStat, ImageEnhance

path = "chest_xray"
outpath = "chest-xray-224"
path = "state-farm-distracted-driver-detection/train"
outpath = "state-farm-distracted-driver-detection/train-224"

def reduce(path, size = 224, method = Image.LANCZOS):
    print(f"Open {path}/db.json")
    with open(f"{path}/db.json", "r") as f:
        db = json.loads(f.read())
    db["path"] = outpath
    for item in db["data"]:
        file = f"{item['path']}/{item['name']}"
        im = Image.open(file)
        if "small" not in item["path"]:
            im = im.resize((size,size),method)
            item["size"] = (size, size)
        item["path"] = item["path"].replace(path, outpath)
        file = file.replace(path, outpath)
        print(f"Create {file}")
        im.save(file)
    print(f"Create {outpath}/db.json")
    with open(f"{outpath}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))

if __name__ == '__main__':
    reduce(path)