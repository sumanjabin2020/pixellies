"""Step 2: training. Tunes each model with 5-fold cross-validation, then saves it."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

df = pd.read_csv("features.csv")
X, y = df.drop(columns=["label", "file"]), df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

searches = {
    "logistic_regression": GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                                        {"logisticregression__C": [0.1, 1, 10, 100]}, cv=5, n_jobs=-1),
    "decision_tree": GridSearchCV(DecisionTreeClassifier(random_state=42),
                                  {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [1, 5, 10]}, cv=5, n_jobs=-1),
    "random_forest": GridSearchCV(RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
                                  {"min_samples_leaf": [1, 3]}, cv=3),
    "gradient_boosting": GridSearchCV(HistGradientBoostingClassifier(max_iter=300, random_state=42),
                                      {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}, cv=3),
}
for name, search in searches.items():
    search.fit(X_train, y_train)
    print(f"{name}: cross-val accuracy = {search.best_score_:.3f}  best = {search.best_params_}")
    joblib.dump(search.best_estimator_, f"models/{name}.joblib")

X_test.assign(label=y_test).to_csv("models/test_split.csv", index=False)
print(f"trained on {len(X_train)} images, held out {len(X_test)} for testing")
