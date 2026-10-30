# Technical Approach Report — Revio Insight Engine

**Candidate:** Nikunj Khandelwal · **Role:** AI/ML Engineer Intern · **Date:** 30 Oct 2026

---

## 1. Goal

Build an ML pipeline that converts raw multi-location reviews into:

1. **Aspect-level sentiment** (what customers like/dislike),
2. **Complaint themes** (recurring problems in plain language),
3. **Crisis alerts** (abnormal negative spikes or high-severity incidents per location),
4. **Grounded, brand-safe reply drafts** with human-in-the-loop approval.

---

## 2. System Architecture

```
 ┌──────────────┐   ┌──────────────┐   ┌───────────────────────────────────────────┐
 │ Review       │   │ Ingestion &  │   │              Enrichment workers            │
 │ sources      │──▶│ Normalisation│──▶│ 1. Language detect + PII scrub            │
 │ (Google, FB, │   │ (dedupe,     │   │ 2. Overall sentiment (fine-tuned RoBERTa) │
 │ industry,    │   │ schema,      │   │ 3. Aspect extraction (LLM → JSON schema)  │
 │ CSV upload)  │   │ location map)│   │ 4. Severity classifier (safety/legal/…)   │
 └──────────────┘   └──────────────┘   │ 5. Embedding (MiniLM / multilingual-E5)   │
                                       └───────────────┬───────────────────────────┘
                                                       ▼
                    ┌───────────────────────┐   ┌───────────────────────────┐
                    │ Postgres + pgvector   │◀──│ Batch analytics (hourly)  │
                    │ reviews, aspects,     │   │ • Theme clustering        │
                    │ embeddings, alerts    │   │   (UMAP + HDBSCAN/BERTopic)│
                    └──────────┬────────────┘   │ • Crisis detector (EWMA / │
                               │                │   z-score per location)   │
                               ▼                └───────────────────────────┘
          ┌────────────────────────────────────────────────────────────┐
          │ Reply service: brand voice + location facts (RAG) → LLM    │
          │ → guardrail checks → auto-post (safe) / human queue (risky)│
          └────────────────────────────────────────────────────────────┘
                               ▼
              Revio dashboard · alerts (email/WhatsApp) · weekly digest
```

**PoC scope (this repo):** CSV ingestion → enrichment → clustering → crisis detection → reply drafting → Streamlit dashboard. Production pieces (queues, Postgres, platform APIs) are described but stubbed.

---

## 3. Data Pipeline

### 3.1 Data sources

| Stage | Data |
|---|---|
| PoC (built) | **Synthetic multi-outlet dataset** (`data/generate_data.py`): 3,136 reviews, 6 outlets of a fictional Indian restaurant chain, 120 days, ~17% Hinglish, PII and cross-posted duplicates included. Two injected events: a food-safety crisis at Delhi CP and a gradual delivery decline at Pune. Chosen over Yelp because Yelp has no Indian multi-outlet chains and no aspect labels, so nothing could be measured against ground truth. |
| Evaluation (built) | Generator ground truth (aspects, severity, event dates) + **40 hand-written challenge reviews** with unseen vocabulary, slang and Hinglish (`data/challenge_set.json`) for an honest generalisation score |
| Next step | Yelp Open Dataset / real Revio pilot data with ~300–1,000 human-labelled reviews |
| Production | Revio's synced reviews (Google Business Profile API etc.), brand voice guide, per-location facts (hours, menu, policies) |

### 3.2 Processing steps

1. **Normalise** to a common schema: `review_id, platform, location_id, rating, text, created_at, language`.
2. **Deduplicate** (hash of text + location + date) — cross-posted reviews are common.
3. **PII scrub** (names, phone numbers, emails) with regex + spaCy NER before any external LLM call.
4. **Language detection** (`fasttext lid.176` / `langdetect`); route Hindi/Hinglish to a multilingual model.
5. **Enrichment** (§4), stored alongside the review.
6. **Aggregation**: daily per-location stats — volume, mean rating, % negative, aspect sentiment shares.

---

## 4. Model Choices

### 4.1 What the PoC actually runs (CPU-only, free, deploys on Streamlit Cloud)

| Task | Implemented in PoC | File |
|---|---|---|
| Overall sentiment | TF-IDF (word 1–2-grams + char 3–5-grams for Hinglish/typos) + Logistic Regression, weak labels from stars, **out-of-fold** predictions so no review is scored by a model that saw it | `src/sentiment.py` |
| Aspects + polarity | Clause splitting + aspect lexicon + polarity lexicon (offline default) **or** Claude with strict JSON schema + evidence-quote hallucination guard | `src/aspects.py` |
| Severity | Rule-based categories (health/safety, harassment, fraud/payment, legal) | `src/aspects.py` |
| Themes | TF-IDF → LSA (TruncatedSVD) → K-Means, k picked by silhouette | `src/themes.py` |
| Crisis | Binomial z-test on 3-day negative share vs 28-day baseline + min uplift, OR ≥2 high-severity reviews in 3 days | `src/crisis.py` |
| Replies | Template drafts (offline) or Claude drafts → guardrails → auto-post / human queue | `src/replies.py` |

### 4.2 Target production models

| Task | Production model | Why | Alternatives considered |
|---|---|---|---|
| Overall sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` (zero-shot), benchmarked against a **TF-IDF + Logistic Regression** baseline trained on star-derived labels | Strong on short, informal text; cheap on CPU; baseline proves the transformer earns its cost | Star rating alone (misses mixed reviews), LLM (too costly per review) |
| Aspect extraction + aspect sentiment | **LLM with strict JSON schema** over a fixed taxonomy (service, staff, wait_time, price, food_quality, cleanliness, delivery, ambience, app_website, safety) | No labelled ABSA data needed to start; handles nuance ("food great but waited 40 min") | PyABSA / DeBERTa-ABSA — planned as a **distilled** model once LLM labels + human corrections reach ~5k reviews |
| Severity classification | LLM label (`none/low/high`, category) + keyword prior | High-recall needed for safety/legal issues; keyword prior guards against LLM misses | Fine-tuned classifier in v1 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (EN); `intfloat/multilingual-e5-small` (Hinglish) | Fast, 384-dim, good clustering quality | OpenAI/Voyage embeddings (cost, data residency) |
| Theme clustering | **BERTopic** (UMAP + HDBSCAN + c-TF-IDF), cluster labels named by LLM | No need to pre-set #topics; produces human-readable themes | K-Means (needs k; spherical clusters) |
| Crisis detection | **Binomial z-test** on rolling negative share per location + severity override | Interpretable, no training data required, accounts for low review counts, tunable per brand | Prophet / isolation forest (later, with more history) |
| Reply generation | Claude (`claude-opus-5-5` by default, configurable via `CLAUDE_MODEL`; a smaller model for routine replies once evaluated) | Quality + instruction-following; brand voice via prompt | Templates (generic), fine-tuned small LLM (later) |

**Cost control:** run cheap models (RoBERTa, embeddings) on every review; call the LLM only for aspect extraction (batched, 10–20 reviews per call) and reply drafting. Cache by review hash.

---

## 5. Prompt Engineering

### 5.1 Aspect extraction prompt (structured output)

```text
SYSTEM: You are an analyst extracting customer feedback from reviews.
Only use these aspects: service, staff, wait_time, price, food_quality,
cleanliness, delivery, ambience, app_website, safety.
Return JSON only, matching the schema. If an aspect isn't mentioned, omit it.
Mark severity "high" ONLY for: health/food-safety, harassment/discrimination,
fraud/payment issues, legal threats, physical injury.

USER: Reviews:
[{"id": "r1", "text": "..."}, {"id": "r2", "text": "..."}]

SCHEMA:
{"results": [{"id": str,
  "aspects": [{"aspect": enum, "sentiment": "positive|negative|neutral",
               "evidence": str /* exact quote from review */}],
  "severity": "none|low|high", "severity_category": str|null}]}
```

Techniques: fixed taxonomy (stable analytics), **evidence quotes** (traceability and hallucination check — evidence must be a substring of the review), few-shot examples for mixed reviews and sarcasm, batching, JSON-schema validation with one automatic retry on parse failure.

### 5.2 Reply generation prompt

```text
SYSTEM: You write replies to customer reviews for {brand_name}.
Voice: {brand_voice_guide}. Max 80 words. Thank the customer, address
the specific points they raised, and invite them to continue offline
at {support_contact} if negative.
RULES: Never admit legal liability. Never promise refunds/compensation.
Never invent facts — use only LOCATION FACTS. Never include personal data.

LOCATION FACTS: {hours, policies, recent_fixes}
REVIEW (rating {stars}): {text}
DETECTED ASPECTS: {aspects_json}
```

### 5.3 Guardrails (post-generation)

| Check | Method | Action on fail |
|---|---|---|
| Forbidden commitments ("refund", "we're liable", "compensate") | Regex + LLM judge | Regenerate once → human queue |
| Ungrounded facts | Compare named entities / numbers with location facts | Human queue |
| PII leakage | Regex/NER | Block |
| Severity = high or rating ≤ 2 | Rule | Always human approval |
| Length/tone | Word count + sentiment of reply | Regenerate |

---

## 6. Evaluation Plan

| Component | Metric | PoC target |
|---|---|---|
| Overall sentiment | Macro-F1 vs star-derived labels | RoBERTa ≥ baseline + 5 pts |
| Aspect extraction | Micro-F1 (aspect + polarity) on 300-review gold set; evidence-substring rate | F1 ≥ 0.75; evidence valid ≥ 95% |
| Severity | Recall on high-severity class | ≥ 0.90 (recall over precision) |
| Themes | Topic coherence (c_v) + manual review of top-10 clusters | Coherent, non-duplicate labels |
| Crisis detector | Precision/recall on injected crisis windows; detection delay (days) | Recall ≥ 0.9, delay ≤ 1 day |
| Replies | Guardrail pass-rate; LLM-as-judge (relevance, tone, groundedness 1–5) + manual spot check | Pass ≥ 95%; mean judge ≥ 4 |

Online (pilot): reply acceptance-without-edit rate, manager time saved, alert usefulness rating, cost per 1k reviews.

### 6.1 PoC results (`python eval/run_eval.py`, full output in `eval/results.md`)

| Component | Result | Comment |
|---|---|---|
| Sentiment (held-out 25%) | Macro-F1 **0.771** vs lexicon baseline 0.736 | Neutral class is hardest (3★ reviews are genuinely mixed) |
| Aspects, in-distribution | F1 0.999 | Optimistic: lexicon matches generator phrasing |
| Aspects, **challenge set** (unseen phrasing) | Precision 0.83, **recall 0.11** | Honest number. Lexicons don't generalise, so production needs the LLM extractor and later a distilled model |
| Severity, in-distribution | Recall 1.00, precision 1.00 | Challenge-set high-severity recall 0.57, same generalisation gap |
| Crisis detector | Injected Delhi crisis caught on **day 0**; **1** false alarm in 720 outlet-days | Thresholds tunable per brand |
| Rating drivers | Pune delivery net sentiment **−1.06** over the last 30 days vs the previous 30 | Detected the injected slow decline, which a spike detector would miss |
| Guardrails | 5/5 adversarial unsafe replies blocked; 18.5% of reviews routed to a human | |

---

## 7. Proof of Concept — Repository Layout

```
cynbit-assessment/
├── README.md · DEMO_GUIDE.md
├── reports/                     # 01_AI_Opportunity_Report.md, 02_Technical_Approach.md
├── data/
│   ├── generate_data.py         # synthetic multi-outlet reviews + injected events
│   ├── sample_reviews.csv       # 3,136 reviews, 6 outlets, 120 days
│   ├── injected_events.json     # ground truth for crisis/drift evaluation
│   ├── challenge_set.json       # 40 hand-written reviews (unseen phrasing)
│   └── brand_profile.json       # voice guide + outlet facts
├── src/
│   ├── ingest.py                # normalise, dedupe, PII scrub, language detect
│   ├── sentiment.py             # TF-IDF + LogReg (out-of-fold) + lexicon baseline
│   ├── aspects.py               # offline ABSA + Claude ABSA with evidence guard
│   ├── themes.py                # TF-IDF -> LSA -> K-Means
│   ├── crisis.py                # binomial z-test + severity override
│   ├── replies.py               # drafts + guardrails + routing
│   ├── prompts.py · llm.py · pipeline.py · config.py
├── app/streamlit_app.py         # 6-tab dashboard
├── eval/run_eval.py             # writes eval/results.md
└── requirements.txt · .env.example
```

**Demo flow:** overview (trend + crisis bands + aspect heatmap) → "why ratings move" for Pune (delivery decline) → complaint themes → crisis alert at Delhi CP → reply studio (draft, guardrails, human routing) → paste a live Hinglish review.

The app runs **fully offline** by default (lexicon extractor + template replies), so reviewers can run it without an API key. Setting `ANTHROPIC_API_KEY` enables Claude.

---

## 8. Production Architecture & MLOps (v1)

- **Serving:** FastAPI microservice; enrichment via a queue (Redis/SQS) so review sync is never blocked; Docker on AWS ECS or Azure Container Apps (matches Cynbit's cloud stack).
- **Storage:** Postgres + pgvector; S3/Blob for raw payloads.
- **Monitoring:** LLM cost & latency per tenant, JSON parse-failure rate, guardrail fail-rate, drift in aspect distribution, alert precision from user feedback ("useful / not useful").
- **Feedback loop:** manager edits to replies and corrections to aspects are stored → periodic fine-tuning of a small ABSA model to replace most LLM calls.
- **Security & compliance:** tenant isolation, RBAC (admin / brand manager / branch advisor — matching Revio's roles), PII scrubbing, DPDP Act-aligned retention.

---

## 9. Execution Plan

| Week | Deliverable | Owner |
|---|---|---|
| 0 | PoC + reports (this submission) | Intern |
| 1 | Label 1–2k pilot reviews; finalise taxonomy with Revio PM | ML + PM |
| 2 | Enrichment service behind Revio sync; store results | ML + Backend |
| 3 | Theme clustering + crisis detector; alert channel | ML |
| 4 | Reply workflow with approval queue in Revio UI | Backend + Frontend |
| 5–6 | Pilot with 2–3 clients; collect feedback; tune thresholds | Team |
| 7 | Monitoring dashboards, cost optimisation, batching | ML / DevOps |
| 8 | v1 GA as Revio premium feature; plan ABSA distillation & Hindi support | Team |

**Success criteria for go/no-go after pilot:** reply acceptance ≥ 70%, crisis alerts rated useful ≥ 80%, LLM cost per 1k reviews within the premium-tier margin.

---

## 10. Limitations & Future Work

- PoC data is synthetic, so in-distribution scores are optimistic. The challenge set shows the real generalisation gap, and real pilot data with human labels is the first next step.
- Statistical crisis detection needs ~4 weeks of history per location to be stable.
- LLM-labelled aspects can be biased by prompt wording — mitigated by the gold set and human corrections.
- Future: competitor benchmarking, review-response impact on rating recovery (causal analysis), voice-of-customer summaries feeding Cynbit's Digital Marketing service.
