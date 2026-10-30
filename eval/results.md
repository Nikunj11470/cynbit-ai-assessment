# Evaluation results

Dataset: 3110 reviews after de-duplication (26 duplicates removed), 6 outlets, 120 days.

## 1. Overall sentiment (25% held-out test set)

| Model | Macro-F1 |
|---|---|
| Lexicon baseline | 0.736 |
| TF-IDF (word+char) + Logistic Regression | **0.771** |

```
              precision    recall  f1-score   support

    negative      0.830     0.824     0.827       148
     neutral      0.519     0.632     0.569       133
    positive      0.945     0.891     0.917       497

    accuracy                          0.834       778
   macro avg      0.764     0.782     0.771       778
weighted avg      0.850     0.834     0.841       778

```

## 2. Aspect + polarity extraction (offline extractor, all reviews)

Micro precision **0.998** · recall **1.000** · F1 **0.999**

| Aspect | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| food_quality | 1.000 | 1.000 | 1.000 | 1903 |
| service | 1.000 | 1.000 | 1.000 | 975 |
| staff | 1.000 | 1.000 | 1.000 | 757 |
| wait_time | 1.000 | 1.000 | 1.000 | 772 |
| price | 0.972 | 1.000 | 0.986 | 597 |
| cleanliness | 1.000 | 1.000 | 1.000 | 427 |
| delivery | 1.000 | 1.000 | 1.000 | 662 |
| ambience | 1.000 | 1.000 | 1.000 | 587 |
| app_website | 1.000 | 1.000 | 1.000 | 192 |
| safety | 1.000 | 1.000 | 1.000 | 61 |

> ⚠️ This score is optimistic: the lexicon was written for the same phrasing the data generator uses. The challenge set below is the honest number.

## 2b. Challenge set (40 hand-written reviews, unseen phrasing)

| Extractor | Precision | Recall | F1 | High-severity recall |
|---|---|---|---|---|
| Offline lexicon | 0.833 | 0.114 | 0.200 | 0.571 |
| Claude | run `python eval/run_eval.py --llm 40` with an API key | | | |

Takeaway: the free offline extractor is fine for known phrasing, but recall drops on new vocabulary - which is why production uses the LLM extractor and later distils its labels into a small fine-tuned model.

## 3. High-severity detection

Recall **1.000** · precision **1.000** (61 true high-severity reviews)

## 4. Crisis early-warning

Injected crisis: DEL-CP food-safety incident 2026-10-18 → 2026-10-27.

- Detected: **yes**, delay **0 day(s)** after onset
- Other alerts (false alarms): **1** over 720 outlet-days (0.14% of outlet-days)

```
location_id      start        end  days  peak_neg_share  high_severity_reviews                                                                                trigger
     DEL-CP 2026-10-18 2026-10-29    12           0.667                     52 22 high-severity reviews in 3 days; negative share spiked to 67% (baseline 27%, z=5.6)
     DEL-CP 2026-08-30 2026-09-01     3           0.529                      1                                     negative share spiked to 53% (baseline 12%, z=5.2)
```

## 5. Reply drafting + guardrails

- Template drafts passing all guardrails: **100.0%** (200 reviews)
- Routed to human approval (≤2★ or high severity): **18.5%**
- Adversarial unsafe replies caught by guardrails: **5/5**
