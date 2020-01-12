import os
import json

path = f"chest_xray"
db = {"path":path, "lum":-1,"std":-1,"lummin":-1,"lummax":-1, "count":0, "data":[] }
id = 0

def scan(path):
    global id
    items = os.listdir(path)
    for item in items:
        if os.path.isdir(f"{path}/{item}"):
            scan(f"{path}/{item}")
        else:
            row = {"id":id, "name":"", "path":"", "lum":-1,"std":-1,"lummin":-1,"lummax":-1}
            row["name"] = item
            row["path"] = path
            db["data"].append(row)
            id+=1

if __name__ == '__main__':
    print(f"Scan {path}")
    scan(path)
    db["count"] = len(db["data"])
    print(f"Create db.json with {db['count']} items")
    with open(f"{path}/db.json", "w") as f:
        f.write(json.dumps(db, indent=4))






