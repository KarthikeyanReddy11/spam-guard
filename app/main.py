"""FastAPI backend. Run:  uvicorn app.main:app --reload"""
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "model" / "spam_model.joblib")
metrics = json.loads((ROOT / "model" / "metrics.json").read_text())

tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
vocab = np.array(tfidf.get_feature_names_out())

app = FastAPI(title="SpamGuard", description="Spam message detector (TF-IDF + Logistic Regression)")


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    return metrics


@app.post("/api/predict")
def predict(msg: Message):
    vec = tfidf.transform([msg.text])
    spam_prob = float(clf.predict_proba(vec)[0][1])

    # Explainability: which words pushed the score toward spam?
    contrib = vec.toarray()[0] * clf.coef_[0]
    top_idx = np.argsort(contrib)[::-1][:5]
    suspicious = [vocab[i] for i in top_idx if contrib[i] > 0]

    return {
        "label": "spam" if spam_prob >= 0.5 else "not spam",
        "spam_probability": round(spam_prob, 4),
        "suspicious_words": suspicious,
    }


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")
