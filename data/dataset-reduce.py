import json
import numpy as np
from PIL import Image, ImageStat, ImageEnhance

path = "chest_xray"
outpath = "chest-xray-224"

def transform(s="train", size = 224, method = Image.LANCZOS):
    print(f"Open {s}/db.json")
    db = None
    with open(f"{path}/{s}/db.json", "r") as f:
        db = json.loads(f.read())
    print(db)
    db["name"] = f"04-center-c{contrast}-{size}-{s}" if center else f"04-norm-c{contrast}-{size}-{s}"
    db["path"] = f"{outpath}/center-c{contrast}-{size}/{s}" if center else f"{outpath}/c{contrast}-{size}/{s}"
    db["size"] = size
    for die in db["dies"]:
        im = Image.open(die["path"])
        stat = ImageStat.Stat(im)
        lum = stat.mean[0]
        norm = im.point(lambda x: np.clip(x - lum + 127, 0, 255))
        if contrast != 1:
            ie = ImageEnhance.Contrast(norm)
            norm = ie.enhance(contrast)
        # stat = ImageStat.Stat(norm)
        # die["lum"] = round(stat.mean[0],2)
        # die["std"] = round(stat.stddev[0],2)
        # die["lummin"] = stat.extrema[0][0]
        # die["lummax"] = stat.extrema[0][1]
        if center:
            radius = config.radiusCenter
            norm = norm.crop(((norm.size[0] / 2) - radius, (norm.size[1] / 2) - radius, (norm.size[0] / 2) + radius, (norm.size[1] / 2) + radius))
        red = norm.resize((size,size),method)
        die["path"] = f"{db['path']}/die{die['id']}-{die['die']}-norm-c{contrast}-{size}.bmp"
        print(f"Creating {die['path']}")
        red.save(die['path'])

        radius = 400
        # im0 = im.crop(((im.size[0] / 2) - radius, (im.size[1] / 2) - radius, (im.size[0] / 2) + radius, (im.size[1] / 2) + radius))
        # im0 = im0.resize((100, 100), Image.LANCZOS)
        # ie = ImageEnhance.Contrast(im0)
        # im0 = ie.enhance(40)
        # pxs = list(im0.getdata())
        # nb0px = len([p for p in pxs if p == 0])
        # die["zpc40"] = nb0px

    # db["lum"] = round(sum([d["lum"] for d in db["dies"]]) / len(db["dies"]), 2)
    # db["std"] = round(sum([d["std"] for d in db["dies"]]) / len(db["dies"]), 2)
    # db["lummin"] = round(sum([d["lummin"] for d in db["dies"]]) / len(db["dies"]), 2)
    # db["lummax"] = round(sum([d["lummax"] for d in db["dies"]]) / len(db["dies"]), 2)
    # db["zpc40"] = int(sum([d["zpc40"] for d in db["dies"]]) / len(db["dies"]))

    print(f"Create {db['path']}/db.json")
    with open(f"{db['path']}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))

# transform("test",1,224)
# transform("train",1,224)
transform("test",4,224)
transform("train",4,224)
# transform("test",6,224)
# transform("train",6,224)
# transform("test",10,224)
# transform("train",10,224)
# transform("test",6,224,center=True)
# transform("train",6,224,center=True)
# transform("test",6,28)
# transform("train",6,28)