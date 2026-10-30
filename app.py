"""Web page for trying the trained models on one image:  python app.py"""
import base64
import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from features import prepare, extract_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
import os
FINAL_FROM = "deep"        # "deep" = final verdict comes from the CLIP model when it exists, "average" = mean of all models
deep_model = embedder = None
if os.path.exists("models/deep_logreg.joblib"):
    import deep_features
    deep_model = joblib.load("models/deep_logreg.joblib")
    embedder = deep_features.load_clip()
models = {"Logistic Regression": joblib.load("models/logistic_regression.joblib"),
          "Decision Tree": joblib.load("models/decision_tree.joblib"),
          "Random Forest": joblib.load("models/random_forest.joblib"),
          "Gradient Boosting": joblib.load("models/gradient_boosting.joblib")}


@app.route("/", methods=["GET", "POST"])
def index():
    result, overall, preview, error = None, None, None, None
    if request.method == "POST":
        data = request.files.get("image")
        raw = data.read() if data else b""
        try:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        except cv2.error:
            img = None
        if img is None:
            error = "That file couldn't be read as an image. Try a JPG or PNG."
        else:
            preview = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
            row = pd.DataFrame([extract_features(prepare(img))])
            result, probs = [], []
            for name, model in models.items():
                p = float(model.predict_proba(row)[0, 1])
                probs.append(p)
                result.append({"model": name, "verdict": "AI-generated" if p >= 0.5 else "Real photo",
                               "confidence": round(max(p, 1 - p) * 100)})
            if deep_model is not None:
                vec = deep_features.embed(embedder, [prepare(img)])
                pd_ = float(deep_model.predict_proba(vec)[0, 1])
                probs.append(pd_)
                result.append({"model": "CLIP + Logistic Regression", "verdict": "AI-generated" if pd_ >= 0.5 else "Real photo",
                               "confidence": round(max(pd_, 1 - pd_) * 100)})
            avg = probs[-1] if (deep_model is not None and FINAL_FROM == "deep") else sum(probs) / len(probs)
            overall = {"ai": avg >= 0.5, "confidence": round(max(avg, 1 - avg) * 100)}
    return render_template("index.html", result=result, overall=overall, preview=preview, error=error)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
