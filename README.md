# Revio Insight Engine — AI opportunity PoC for Cynbit

**Candidate:** Nikunj Khandelwal · **Role:** AI/ML Engineer Intern · **Assessment:** Identify an AI Opportunity for Cynbit

An ML layer for **Revio**, Cynbit's AI reputation-management product. It turns raw multi-location reviews into:

1. **Aspect-level sentiment**: what guests like and dislike (food, staff, wait time, delivery…), with the exact quote as evidence
2. **"Why did the rating move?"**: per-outlet drivers of rating change
3. **Complaint themes**: recurring problems discovered by clustering
4. **Crisis early-warning**: statistical spike detection plus a high-severity override (food safety, harassment, fraud, legal)
5. **Brand-safe reply drafts**: grounded in outlet facts, checked by guardrails, routed to a human when risky

> 🔗 **Live demo:** https://cynbit-revio-insight.streamlit.app/ (first load takes ~1–2 min while the pipeline runs)

## Deliverables

| Deliverable | Where |
|---|---|
| AI Opportunity Report | [`reports/01_AI_Opportunity_Report.md`](reports/01_AI_Opportunity_Report.md) |
| Technical Approach Report | [`reports/02_Technical_Approach.md`](reports/02_Technical_Approach.md) |
| Proof of Concept (Streamlit app) | [`app/streamlit_app.py`](app/streamlit_app.py) + [`src/`](src/) |
| Evaluation results | [`eval/results.md`](eval/results.md) |
| Demo script | [`DEMO_GUIDE.md`](DEMO_GUIDE.md) |

## Quick start

```bash
pip install -r requirements.txt
python data/generate_data.py        # optional: the dataset is already committed
streamlit run app/streamlit_app.py  # opens http://localhost:8501
python eval/run_eval.py             # prints metrics and writes eval/results.md
```

The app runs **fully offline** by default; no API key is needed. To enable Claude for reply drafting and the LLM aspect extractor:

```bash
cp .env.example .env   # then put your key in it, or: export ANTHROPIC_API_KEY=...
python eval/run_eval.py --llm 40    # scores Claude on the hand-written challenge set
```

## Architecture

```
reviews CSV ─▶ ingest (dedupe, PII scrub, language) ─▶ sentiment model (TF-IDF + LogReg, out-of-fold)
                                                   └─▶ aspect extractor (offline lexicon | Claude JSON schema)
                                                          ├─▶ severity tagging
                                                          ├─▶ themes (TF-IDF → LSA → K-Means, silhouette-selected k)
                                                          └─▶ crisis detector (binomial z-test + severity override)
                                                                 └─▶ reply drafting → guardrails → auto-post / human queue
```

| Folder | Contents |
|---|---|
| `data/` | `generate_data.py` (synthetic multi-outlet reviews with injected events), `sample_reviews.csv`, `brand_profile.json`, `challenge_set.json` (40 hand-written reviews for honest evaluation) |
| `src/` | `ingest.py`, `sentiment.py`, `aspects.py`, `themes.py`, `crisis.py`, `replies.py`, `prompts.py`, `llm.py`, `pipeline.py` |
| `app/` | Streamlit dashboard (6 tabs) |
| `eval/` | `run_eval.py` and the generated `results.md` |

## Key results (offline, from `eval/results.md`)

| Component | Result |
|---|---|
| Overall sentiment, held-out test | Macro-F1 **0.771** (lexicon baseline 0.736) |
| Aspect extraction, in-distribution | F1 0.999 *(optimistic: lexicon matches the generator's phrasing)* |
| Aspect extraction, **unseen phrasing** (challenge set) | Precision 0.83 / **Recall 0.11**: this is why production uses the LLM extractor |
| Crisis detection | Injected food-safety crisis detected on **day 0**; 1 false alarm in 720 outlet-days |
| Reply guardrails | 5/5 adversarial unsafe replies blocked; 18.5% of reviews routed to a human |

## Why synthetic data?

Revio's client reviews are private, and public datasets have no Indian multi-outlet chains with aspect labels. The generator gives **known ground truth** (aspects, severity, an injected crisis at Delhi CP, a slow delivery decline at Pune) so every component can be measured. It also covers Hinglish, cross-posted duplicates and PII. The brand **"Tadka Lane" is fictional**. Swapping in real data only requires a CSV with `review_id, location_id, location_name, platform, created_at, rating, text`.

## Deploy (free) on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to share.streamlit.io → **New app** → select the repo, branch `main`, main file `app/streamlit_app.py`.
3. (Optional) **Advanced settings → Secrets**: `ANTHROPIC_API_KEY = "sk-ant-..."`
4. Deploy and paste the URL at the top of this README.
