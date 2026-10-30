"""
OPTIONAL EXTENSION: trains Logistic Regression on the CLIP features and compares it with the
handcrafted models on exactly the same held-out test images.

    python train_deep.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

df = pd.read_csv("features.csv")
y = df["label"]
X_hand = df.drop(columns=["label", "file"])
X_deep = np.load("deep_features.npy")
assert len(X_deep) == len(df), "run deep_features.py again, the image set changed"

# same split as train.py (same seed, same stratification) so the numbers are comparable
idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=42)

search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                      {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=5, n_jobs=-1)
search.fit(X_deep[idx_train], y.iloc[idx_train])
print(f"deep logistic regression: cross-val accuracy = {search.best_score_:.3f}  best = {search.best_params_}")
joblib.dump(search.best_estimator_, "models/deep_logreg.joblib")

y_test = y.iloc[idx_test]
probs = {"CLIP + Logistic Regression": search.best_estimator_.predict_proba(X_deep[idx_test])[:, 1]}
hand = {}
for key, title in [("logistic_regression", "Logistic Regression"), ("decision_tree", "Decision Tree"),
                   ("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting")]:
    try:
        hand[title] = joblib.load(f"models/{key}.joblib").predict_proba(X_hand.iloc[idx_test])[:, 1]
    except FileNotFoundError:
        pass
if hand:
    avg_hand = sum(hand.values()) / len(hand)
    probs["Handcrafted average"] = avg_hand
    probs["Deep + handcrafted"] = (probs["CLIP + Logistic Regression"] * 2 + avg_hand) / 3

print(f"\n{'model':<28}{'accuracy':>10}{'precision':>11}{'recall':>9}{'roc-auc':>9}")
for title, p in {**hand, **probs}.items():
    pred = p >= 0.5
    print(f"{title:<28}{accuracy_score(y_test, pred):>10.3f}{precision_score(y_test, pred):>11.3f}"
          f"{recall_score(y_test, pred):>9.3f}{roc_auc_score(y_test, p):>9.3f}")
