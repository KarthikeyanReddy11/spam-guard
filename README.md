# SpamGuard: Spam Message Detector (ML + FastAPI + Web UI)

A full-stack machine learning project. You type a message, the model says **spam / not spam**, shows the **probability**, and highlights the **words that raised suspicion**.

**Result:** 97.6% accuracy on 1,115 messages the model never saw during training (spam recall 92.6%).

## Memory trick: "D-T-S-A" = the 4 steps of every ML project
**Data → Train → Serve → App**

| Step | File | Pizza-shop version |
|---|---|---|
| Data | `data/sms.tsv` | Raw ingredients (5,574 labelled SMS) |
| Train | `train.py` | Cooking the pizza (learning from data) |
| Serve | `app/main.py` | Delivery counter (API that answers requests) |
| App | `static/index.html` | The menu customers see (web page) |

## Run it on your laptop
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                    # trains the model (takes a few seconds)
uvicorn app.main:app --reload      # open http://127.0.0.1:8000
```
API docs are automatic at `/docs`. Tests: `pip install pytest httpx && pytest -q`.

## Deploy it (free) so you have a live link
1. Create a GitHub repo and push this folder.
2. Go to **render.com** → New → **Web Service** → connect your repo.
3. Choose **Docker** as the runtime (the `Dockerfile` is ready). Pick the free plan.
4. Deploy. You get a public URL like `https://spamguard-xyz.onrender.com`.
(The free plan sleeps after inactivity, so the first load can take ~30 seconds.)

## How it works (explain it like this)
1. **TF-IDF** converts each message into numbers. A word gets a high score if it is frequent in *this* message but rare across *all* messages. ("prize" scores high, "the" scores low.)
2. **Logistic Regression** learns a weight for each word. Positive weight means it pushes toward spam. It outputs a probability between 0 and 1.
3. The **80/20 split**: the model learns on 80% and is tested on the hidden 20%, so the accuracy is honest.
4. The **"suspicious words"** feature multiplies each word's TF-IDF value by its learned weight and shows the top ones. That makes the model explainable.

## Make it YOUR project (do at least 4 of these)
- [ ] Train a second model (Naive Bayes) in `train.py` and compare accuracy in a table.
- [ ] Add a **history** panel on the page showing your last 5 checks.
- [ ] Add a `/api/predict-batch` endpoint that accepts many messages.
- [ ] Add a **feedback button** ("wrong result?") that saves messages to a CSV for retraining.
- [ ] Change the spam threshold (0.5) to 0.7 and record how false alarms vs missed spam change.
- [ ] Draw your own architecture diagram and put it in your report.
- [ ] Commit to GitHub in small steps with meaningful messages (your commit history is proof of your effort).

## Viva questions you should be able to answer
- **Why split into train and test?** To check the model on data it hasn't memorised.
- **Why not just use accuracy?** Only 13% of messages are spam, so a model that always says "not spam" would score 87%. That's why we also check precision and recall.
- **Precision vs recall here?** Precision: of the messages flagged spam, how many really were. Recall: of all real spam, how many we caught.
- **What is `class_weight="balanced"`?** It makes mistakes on the rarer class (spam) cost more.
- **Why FastAPI?** Fast, automatic validation (Pydantic) and automatic docs at `/docs`.
- **Limitations?** Trained on older English SMS data. New scam styles or other languages need retraining.

## Dataset
SMS Spam Collection (UCI / Almeida & Hidalgo), 5,574 labelled messages. Cite it in your report.
