import io, json, time, random
import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from features import prepare

REPO = "julienlucas/midjourney-dalle-sd-dataset"
FILES = ["data/test-00000-of-00001.parquet", "data/train-00000-of-00003.parquet",
         "data/train-00001-of-00003.parquet", "data/train-00002-of-00003.parquet"]
TARGET = 3000          # images per class
random.seed(1)
count = {"real": 0, "ai": 0}
sizes = {"real": {}, "ai": {}}


def degrade(bgr):          # same web-style shrink + JPEG as get_data.py, applied to both classes
    h, w = bgr.shape[:2]
    if random.random() < 0.7:
        f = random.uniform(max(0.5, 300 / min(h, w)), 1.0)
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 95)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def fetch(name):
    for attempt in range(1, 8):
        try:
            return hf_hub_download(REPO, name, repo_type="dataset")
        except Exception as e:
            print(f"  download interrupted ({str(e)[:60]}), retry {attempt}/7"); time.sleep(3)
    raise SystemExit("download keeps failing - run this again, it resumes")


def full():
    return all(v >= TARGET for v in count.values())


for name in FILES:
    if full():
        break
    print(f"\nopening {name} ...")
    pf = pq.ParquetFile(fetch(name))
    names = pf.schema_arrow.names
    lab = next(c for c in names if "label" in c.lower())
    try:
        cls = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"][lab]["names"]
    except Exception:
        cls = None
    for batch in pf.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            v = row[lab]
            label = v if isinstance(v, str) else (cls[v] if cls else None)
            if label is None:
                raise SystemExit("could not tell which label is real - send this message to Claude")
            kind = "real" if "real" in label.lower() else "ai"
            if count[kind] >= TARGET:
                continue
            im = Image.open(io.BytesIO(row["image"]["bytes"]))
            if min(im.size) < 256:
                continue
            sizes[kind][im.size] = sizes[kind].get(im.size, 0) + 1
            bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"data/{kind}/mj_{count[kind]:05d}.png", prepare(degrade(bgr)))
            count[kind] += 1
        print(f"  real {count['real']}/{TARGET}   ai {count['ai']}/{TARGET}", end="\r")
        if full():
            break

print(f"\nsaved {count['real']} real and {count['ai']} AI modern images")
for k in sizes:
    print(k, "top sizes:", sorted(sizes[k].items(), key=lambda x: -x[1])[:3])
