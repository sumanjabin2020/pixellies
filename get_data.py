"""Builds data/real and data/ai from the already-downloaded dataset (no big re-download).
Real photos smaller than MIN_SIDE are skipped, because the AI images are all 1024x1024 and
tiny real photos would let the models cheat by spotting enlargement. Every image then gets
the same treatment (shrink to 256, centre crop), and the two classes are balanced."""
import io, os, glob, time, random, collections
import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from features import prepare

LIMIT = 3333            # AI images available in this dataset
REAL_SAMPLE = 0.8       # sample real images across all parts, so they are not all from one file
MIN_SIDE = 400          # smallest real photo we accept (shorter side, in pixels)
REPO = "hsbharadwaj/AI-vs-Real"
random.seed(42)

for k in ("real", "ai"):
    os.makedirs(f"data/{k}", exist_ok=True)
    for old in glob.glob(f"data/{k}/{k}_?????.png"):      # only our own files; your extra images are kept
        os.remove(old)
count = {"real": 0, "ai": 0}
sizes = {"real": collections.Counter(), "ai": collections.Counter()}
fmts = {"real": collections.Counter(), "ai": collections.Counter()}


def degrade(bgr):
    """Applied to BOTH classes: random shrink + random JPEG quality, like images found on the web.
    This stops the models using size or compression history as a shortcut."""
    h, w = bgr.shape[:2]
    if random.random() < 0.7:
        f = random.uniform(max(0.5, 300 / min(h, w)), 1.0)
        bgr = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, random.randint(55, 95)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def get_part(s):
    for attempt in range(1, 8):
        try:
            return hf_hub_download(REPO, f"data/train-{s:05d}-of-00005.parquet", repo_type="dataset")
        except Exception as e:
            print(f"  download interrupted ({str(e)[:60]}), retry {attempt}/7"); time.sleep(3)
    raise SystemExit("download keeps failing - run this again, it resumes")


for s in range(5):
    print(f"\nopening part {s} ...")
    pf = pq.ParquetFile(get_part(s))
    lab = next(c for c in pf.schema_arrow.names if "label" in c.lower())
    for batch in pf.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            kind = "real" if row[lab] == 1 else "ai"
            if count[kind] >= LIMIT or (kind == "real" and random.random() > REAL_SAMPLE):
                continue
            im = Image.open(io.BytesIO(row["image"]["bytes"]))
            if kind == "real" and min(im.size) < MIN_SIDE:
                continue
            sizes[kind][im.size] += 1
            fmts[kind][im.format] += 1
            bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"data/{kind}/{kind}_{count[kind]:05d}.png", prepare(degrade(bgr)))
            count[kind] += 1
        print(f"  real {count['real']}/{LIMIT}   ai {count['ai']}/{LIMIT}", end="\r")

for a, b in (("real", "ai"), ("ai", "real")):          # balance: trim the bigger class
    extra = count[a] - count[b]
    for i in range(count[a] - 1, count[a] - 1 - max(extra, 0), -1):
        os.remove(f"data/{a}/{a}_{i:05d}.png")
    count[a] -= max(extra, 0)
print(f"\nsaved {count['real']} real and {count['ai']} AI images\n")
for k in ("real", "ai"):
    print(k, "top sizes:", sizes[k].most_common(3), "| formats:", dict(fmts[k]))
