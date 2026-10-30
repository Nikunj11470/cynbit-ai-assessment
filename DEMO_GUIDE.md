# Demo Guide — Revio Insight Engine (5–7 minutes)

Use this for a live demo on a call **or** to record a 5-minute video (Loom / OBS / Windows `Win + Alt + R`) whose link goes in the submission portal as an additional deliverable.

**Before you start:** open the live Streamlit URL (or `streamlit run app/streamlit_app.py`) and wait for the pipeline to finish (~20–30 s on first load). Keep the GitHub repo open in a second tab.

---

## 0:00 — The problem (30 s, no screen needed)

> "Cynbit's Revio already syncs reviews and generates AI replies. But a brand with many outlets still can't answer three questions: **why** is an outlet's rating dropping, **is this spike a crisis**, and **is this AI reply safe to post**? I built an ML layer for Revio that answers all three."

## 0:30 — Overview tab (1 min)

1. Point to the KPI row: 3,111 reviews, 6 outlets of a **fictional** chain, 120 days.
2. Sidebar caption: *"The ingestion step removed 25 cross-posted duplicates, masked phone numbers and emails before anything reaches an LLM, and detected ~500 Hinglish reviews. That matters for Indian clients."*
3. Rating chart: *"The red band is a crisis my detector raised at Delhi Connaught Place."*
4. Heatmap: *"Every review is broken into aspects. Green is a strength, red is a problem. Health & safety is −1 everywhere because it's only ever mentioned negatively."*

## 1:30 — Why ratings move (1 min) ⭐ strongest moment

1. Pick **Pune – Koregaon Park**, keep 30 days.
2. *"Pune's rating is falling. Star averages don't say why. This view compares aspect sentiment for the last 30 days with the 30 before. **Delivery** collapsed (about −1.0) while food and service are flat. A manager now knows to talk to the delivery partner, not the chef."*
3. *"It's a slow decline, so a spike detector would never fire. That's why both views exist."*

## 2:30 — Complaint themes (45 s)

*"Negative snippets are clustered automatically (TF-IDF → LSA → K-Means, k picked by silhouette score). Nobody defined these topics: 'hidden service charge', 'packaging leaked', 'waited an hour despite booking'. Each one shows how often it occurs and which outlet it's concentrated in."*

## 3:15 — Crisis alerts (1 min)

1. Show the incident table. Delhi CP has ~22 high-severity reviews in 3 days.
2. Pick Delhi CP in "Inspect outlet signal": *"The red line is the 3-day negative share, the dashed line is that outlet's own 28-day baseline. I use a **binomial z-test** so a bad day at a quiet outlet doesn't trigger a false alarm. Separately, 2+ food-safety reviews in 3 days always alerts."*
3. Scroll to the latest high-severity reviews (food poisoning, cockroach).
4. *"On my evaluation it caught the injected crisis on day zero, with 1 false alarm in 720 outlet-days."*

## 4:15 — Reply studio (1 min)

1. Pick a 1★ review. *"The draft acknowledges the specific aspects and pulls only facts from the outlet profile. Here it mentions the new delivery packaging, so it can't invent things."*
2. Point at the guardrail checklist: *"No refund or liability promises, no personal data, no invented numbers. Low-rated or high-severity reviews are always routed to a human. AI drafts, a person approves."*
3. (If an API key is set) toggle **Use Claude**: *"Same guardrails apply to the LLM draft, and a failing draft is regenerated once."*

## 5:15 — Try a review (45 s)

Click **Analyse** on the prefilled Hinglish review:
*"It understands 'swad zabardast tha' as positive food quality, flags the wait and the rude waiter, masks the phone number, and drafts a reply."*

Optionally type your own review live. Interviewers love that.

## 6:00 — Honest evaluation + next steps (45 s)

Open `eval/results.md` in GitHub:
> "In-distribution, the offline extractor scores 0.99 F1, but that's optimistic because the data is synthetic. So I hand-wrote 40 reviews with new phrasing: **recall drops to 0.11**. That's exactly why the production design uses Claude for extraction and then distils those labels into a small fine-tuned model to cut cost. Next step: a 4-week pilot on 2–3 real Revio clients."

---

## Likely interview questions (prepare these)

| Question | Answer in one breath |
|---|---|
| Why this opportunity over others? | It upgrades a product Cynbit already sells. It's measurable (F1, alert precision, reply acceptance), demoable with no private data, and the same engine helps the Digital Marketing service (local SEO depends on reviews). |
| Why synthetic data? | Real Revio reviews are private, and public datasets have no Indian multi-outlet chains or aspect labels. Synthetic data gives known ground truth and injected events so every component is measurable. I added a hand-written challenge set so I'm not fooling myself. |
| Why not just use an LLM for everything? | Cost and latency at volume. Cheap models (TF-IDF sentiment, statistics) run on every review; the LLM is used only where it adds value (extraction on new phrasing, reply drafting), batched and cached. Long-term: distil LLM labels into a small model. |
| How do you stop hallucinated replies? | Grounding (outlet facts only), a rule-based guardrail layer, the evidence-quote check on extraction, and human approval for risky reviews. |
| Why a binomial z-test? | Review counts vary a lot per outlet and day. The binomial standard error makes the threshold stricter when there are few reviews, so you get fewer false alarms. It's interpretable for a manager too. |
| Why is the neutral class weak? | 3★ reviews are genuinely mixed; star-derived labels are noisy. Fix: human labels or aspect-level scoring instead of one overall label. |
| How would you deploy it? | FastAPI microservice behind Revio's sync, a queue for enrichment, Postgres + pgvector, Docker on AWS/Azure (Cynbit's stack), plus monitoring of LLM cost, guardrail failures and alert feedback. |
| Privacy? | PII masked before storage/LLM calls, tenant isolation, retention aligned with India's DPDP Act 2023. |

## Submission checklist

- [ ] Repo pushed to GitHub and **public**
- [ ] Streamlit Cloud app deployed; URL added to README
- [ ] Portal: **Primary link** = GitHub repo; **Additional links** = live demo URL, (optional) demo video, (optional) reports as PDF
- [ ] Portal setup notes: *"Run `pip install -r requirements.txt` then `streamlit run app/streamlit_app.py`. Works offline; optional ANTHROPIC_API_KEY enables Claude."*
