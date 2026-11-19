"""Train the spam classifier and save it.
Run:  python train.py
Pipeline = TF-IDF (turns text into numbers) -> Logistic Regression (decides spam / not spam)
"""
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. Load data: each row is "label<TAB>message"
df = pd.read_csv("data/sms.tsv", sep="\t", names=["label", "text"])
df["y"] = (df["label"] == "spam").astype(int)  # spam=1, ham=0

# 2. Split: 80% to learn from, 20% kept hidden to test honestly
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)

# 3. Build the pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)

# 4. Evaluate on the unseen 20%
pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "confusion": {"true_ham": int(tn), "false_spam": int(fp), "missed_spam": int(fn), "caught_spam": int(tp)},
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
}
print(json.dumps(metrics, indent=2))

# 5. Save
joblib.dump(model, "model/spam_model.joblib")
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved model/spam_model.joblib")
