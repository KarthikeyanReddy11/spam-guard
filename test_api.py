"""Run:  pytest -q   (pip install pytest httpx)"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}

def test_spam_detected():
    r = client.post("/api/predict", json={"text": "WINNER!! Claim your free prize now, call 09061701461"})
    assert r.status_code == 200 and r.json()["label"] == "spam"

def test_normal_message():
    r = client.post("/api/predict", json={"text": "Are we meeting at the library at 5?"})
    assert r.json()["label"] == "not spam"

def test_empty_rejected():
    assert client.post("/api/predict", json={"text": ""}).status_code == 422
