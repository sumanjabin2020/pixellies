"""
OPTIONAL EXTENSION (not part of the original abstract): deep features.

A pretrained CLIP image network turns every image into 512 numbers. Those numbers are then
fed to a simple Logistic Regression (see train_deep.py), so the classifier stays classical
but the features come from a network that has seen hundreds of millions of images.

    python deep_features.py      # builds deep_features.npy, same row order as features.csv

First run downloads the CLIP model (about 600 MB, once).
"""
import os
import cv2
import numpy as np
import pandas as pd
from features import prepare

MODEL_NAME = "openai/clip-vit-base-patch32"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def load_clip():
    import torch
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).eval()
    return model


def to_tensor_batch(bgr_images):
    """BGR 256x256 crops -> normalised 224x224 RGB array batch for CLIP."""
    out = []
    for img in bgr_images:
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        out.append(((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
    return np.stack(out)


def embed(model, bgr_images):
    """Returns an (n, 512) array, one L2-normalised vector per image."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(to_tensor_batch(bgr_images))
        v = model(pixel_values=x).image_embeds
        v = v / v.norm(dim=-1, keepdim=True)
    return v.numpy()


def main():
    df = pd.read_csv("features.csv", usecols=["label", "file"])
    folder = {0: "real", 1: "ai"}
    model = load_clip()
    vecs, batch = [], []
    for i, (label, name) in enumerate(zip(df["label"], df["file"])):
        img = cv2.imread(os.path.join("data", folder[label], name))
        batch.append(prepare(img))
        if len(batch) == 32 or i == len(df) - 1:
            vecs.append(embed(model, batch)); batch = []
            print(f"  {i + 1}/{len(df)}", end="\r")
    np.save("deep_features.npy", np.concatenate(vecs))
    print(f"\nsaved deep_features.npy for {len(df)} images")


if __name__ == "__main__":
    main()
