"""
Picks a small, balanced sample from the Kaggle "AI vs Human Generated Dataset"
(the one with train.csv) and copies it into data/real and data/ai.

    python prepare_data.py /path/to/extracted/dataset-folder

The folder must contain train.csv (columns: file_name, label; 1 = AI, 0 = human).
Change PER_CLASS if you want more or fewer images.
"""
import os
import sys
import shutil
import pandas as pd

PER_CLASS = 150

if len(sys.argv) != 2:
    sys.exit("usage: python prepare_data.py /path/to/dataset-folder")

root = sys.argv[1]
df = pd.read_csv(os.path.join(root, "train.csv"))

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/ai", exist_ok=True)

for label, folder in [(0, "real"), (1, "ai")]:
    rows = df[df["label"] == label].sample(min(PER_CLASS * 2, (df["label"] == label).sum()), random_state=42)  # spare ones in case a file is missing
    n = 0
    for name in rows["file_name"]:
        src = os.path.join(root, name)
        if not os.path.exists(src):
            continue
        ext = os.path.splitext(src)[1]
        shutil.copy(src, f"data/{folder}/{folder}_{n:04d}{ext}")
        n += 1
        if n == PER_CLASS:
            break
    print(f"{folder}: copied {n} images")
