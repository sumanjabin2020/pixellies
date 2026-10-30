"""Step 3: evaluation on the held-out test images."""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

test = pd.read_csv("models/test_split.csv")
X_test, y_test = test.drop(columns=["label"]), test["label"]
names = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
         "gradient_boosting": "Gradient Boosting"}

plt.figure(figsize=(5, 4.5))
all_probs = []
print(f"{'model':<22}{'accuracy':>10}{'precision':>11}{'recall':>9}{'roc-auc':>9}")
for key, title in names.items():
    model = joblib.load(f"models/{key}.joblib")
    pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    print(f"{title:<22}{accuracy_score(y_test, pred):>10.3f}{precision_score(y_test, pred):>11.3f}"
          f"{recall_score(y_test, pred):>9.3f}{roc_auc_score(y_test, prob):>9.3f}")
    all_probs.append(prob)
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{title} (AUC {roc_auc_score(y_test, prob):.2f})")
avg = sum(all_probs) / len(all_probs)          # what the website shows as its final answer
print(f"{'Average of all models':<22}{accuracy_score(y_test, avg >= 0.5):>10.3f}{precision_score(y_test, avg >= 0.5):>11.3f}"
      f"{recall_score(y_test, avg >= 0.5):>9.3f}{roc_auc_score(y_test, avg):>9.3f}")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve")
plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig("outputs/roc_curve.png", dpi=150); plt.close()

feats = pd.read_csv("features.csv")
show = ["noise_std", "edge_density", "fft_high_ratio", "laplacian_var"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for ax, col in zip(axes, show):
    ax.hist(feats[feats.label == 0][col], bins=25, alpha=0.6, label="real")
    ax.hist(feats[feats.label == 1][col], bins=25, alpha=0.6, label="AI")
    ax.set_title(col)
axes[0].legend(); plt.tight_layout(); plt.savefig("outputs/feature_distributions.png", dpi=150); plt.close()

rf = joblib.load("models/random_forest.joblib")
imp = pd.Series(rf.feature_importances_, index=X_test.columns).nlargest(15)[::-1]
plt.figure(figsize=(6, 5)); imp.plot.barh(); plt.title("Top 15 features (Random Forest)")
plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=150)
print("saved plots to outputs/")
