# PixelLies

Detecting AI-generated images with handcrafted statistical features and classical ML
(Logistic Regression and Decision Tree). Machine Learning micro-project, S5 CSE.

## Setup

    pip install -r requirements.txt

## Dataset

Easiest way: `pip install datasets pillow` then `python get_data.py` (downloads 150 + 150 images).

Or, if you downloaded the Kaggle "AI vs Human Generated Dataset" (has train.csv): `python prepare_data.py /path/to/folder`.

Or do it by hand:

Put your images in two folders (about 100+ each works fine, keep them equal):

    data/real/   real photographs
    data/ai/     AI-generated images

Sources: Kaggle "CIFAKE" or "AI vs Human generated images", or Unsplash photos for
real ones and any free AI image set for generated ones. Use the same kind of subject
in both folders if you can, so the models learn texture and noise, not content.

## Run, one step at a time

    python features.py    # images -> features.csv   (26 numbers per image)
    python train.py       # features.csv -> models/  (trains both models)
    python evaluate.py    # accuracy, precision, recall, ROC-AUC + plots in outputs/
    python app.py         # small web page, open http://127.0.0.1:5000

## Features (features.py)

- Colour: mean, std, skew and histogram entropy per channel, plus saturation mean/std
- Edges: Canny edge density, Laplacian variance
- Noise: spread of the residual after a median blur
- Frequency: Fourier energy in 6 bands, high-frequency ratio, high-frequency peak

Images are resized to 256x256 and scaled to 0-1 before any feature is computed.
