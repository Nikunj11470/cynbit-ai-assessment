"""Offline evaluation of every component. Writes eval/results.md.

    python eval/run_eval.py            # offline components
    python eval/run_eval.py --llm 40   # also score Claude aspect extraction on the 40-review challenge set
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import aspects as absa  # noqa: E402
from src import crisis  # noqa: E402
from src.config import ASPECTS, CLAUDE_MODEL, EVENTS_JSON  # noqa: E402
from src.ingest import load_reviews  # noqa: E402
from src.replies import generate_reply, guardrail_check, load_brand  # noqa: E402
from src.sentiment import SentimentModel, build_model, lexicon_sentiment, stars_to_label  # noqa: E402


def pair_set(aspects: list[dict]) -> set[tuple[str, str]]:
    return {(a["aspect"], a["sentiment"]) for a in aspects}


def prf(gold: list[set], pred: list[set]) -> tuple[float, float, float]:
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    fp = sum(len(p - g) for g, p in zip(gold, pred))
    fn = sum(len(g - p) for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main(llm_n: int) -> None:
    df, stats = load_reviews()
    out = ["# Evaluation results", "", f"Dataset: {len(df)} reviews after de-duplication "
           f"({stats['duplicates_removed']} duplicates removed), 6 outlets, 120 days.", ""]

    # 1. Overall sentiment: held-out test split, star-derived labels
    y = stars_to_label(df["rating"])
    X_tr, X_te, y_tr, y_te = train_test_split(df["text_clean"], y, test_size=0.25, random_state=42, stratify=y)
    model = build_model().fit(X_tr, y_tr)
    pred_ml = model.predict(X_te)
    pred_lex = X_te.map(lexicon_sentiment)
    f1_ml, f1_lex = f1_score(y_te, pred_ml, average="macro"), f1_score(y_te, pred_lex, average="macro")
    out += ["## 1. Overall sentiment (25% held-out test set)", "",
            "| Model | Macro-F1 |", "|---|---|",
            f"| Lexicon baseline | {f1_lex:.3f} |", f"| TF-IDF (word+char) + Logistic Regression | **{f1_ml:.3f}** |", "",
            "```", classification_report(y_te, pred_ml, digits=3), "```", ""]

    # 2. Aspect extraction vs ground-truth labels
    gold = [pair_set(json.loads(t)) for t in df["true_aspects"]]
    extracted = [absa.extract_offline(t) for t in df["text_clean"]]
    pred = [pair_set(e["aspects"]) for e in extracted]
    p, r, f = prf(gold, pred)
    out += ["## 2. Aspect + polarity extraction (offline extractor, all reviews)", "",
            f"Micro precision **{p:.3f}** · recall **{r:.3f}** · F1 **{f:.3f}**", "",
            "| Aspect | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for asp in ASPECTS:
        g = [{x for x in s if x[0] == asp} for s in gold]
        q = [{x for x in s if x[0] == asp} for s in pred]
        support = sum(len(s) for s in g)
        if support:
            ap, ar, af = prf(g, q)
            out.append(f"| {asp} | {ap:.3f} | {ar:.3f} | {af:.3f} | {support} |")
    out += ["", "> ⚠️ This score is optimistic: the lexicon was written for the same phrasing the data generator uses. "
            "The challenge set below is the honest number.", ""]

    # 2b. Hand-written challenge set: new vocabulary, slang, Hinglish, implicit complaints
    challenge = json.loads((ROOT / "data" / "challenge_set.json").read_text(encoding="utf-8"))
    ch_gold = [{tuple(a) for a in c["aspects"] if a[1] != "neutral"} for c in challenge]
    ch_off = [absa.extract_offline(c["text"]) for c in challenge]
    cp, cr, cf = prf(ch_gold, [pair_set(e["aspects"]) for e in ch_off])
    ch_true_high = [c["severity"] == "high" for c in challenge]
    ch_sev_rec = sum(t and e["severity"] == "high" for t, e in zip(ch_true_high, ch_off)) / sum(ch_true_high)
    out += [f"## 2b. Challenge set ({len(challenge)} hand-written reviews, unseen phrasing)", "",
            "| Extractor | Precision | Recall | F1 | High-severity recall |", "|---|---|---|---|---|",
            f"| Offline lexicon | {cp:.3f} | {cr:.3f} | {cf:.3f} | {ch_sev_rec:.3f} |"]
    if llm_n:
        items = [(f"C{i}", c["text"]) for i, c in enumerate(challenge[:llm_n])]
        res = absa.extract_llm(items)
        q = [{x for x in pair_set(res[rid]["aspects"]) if x[1] != "neutral"} for rid, _ in items]
        lp, lr, lf = prf(ch_gold[:llm_n], q)
        l_sev = sum(t and res[rid]["severity"] == "high" for t, (rid, _) in zip(ch_true_high, items)) / max(sum(ch_true_high[:llm_n]), 1)
        dropped = sum(res[rid].get("evidence_dropped", 0) for rid, _ in items)
        fallbacks = sum(bool(res[rid].get("fallback")) for rid, _ in items)
        out.append(f"| Claude ({CLAUDE_MODEL}) | {lp:.3f} | {lr:.3f} | {lf:.3f} | {l_sev:.3f} |")
        out += ["", f"Claude run: {dropped} evidence quotes rejected by the hallucination guard, {fallbacks} offline fallbacks."]
    else:
        out.append("| Claude | run `python eval/run_eval.py --llm 40` with an API key | | | |")
    out += ["", "Takeaway: the free offline extractor is fine for known phrasing, but recall drops on new vocabulary - "
            "which is why production uses the LLM extractor and later distils its labels into a small fine-tuned model.", ""]

    # 3. Severity
    true_high = df["true_severity"].eq("high")
    pred_high = pd.Series([e["severity"] == "high" for e in extracted], index=df.index)
    tp = int((true_high & pred_high).sum())
    sev_rec = tp / max(int(true_high.sum()), 1)
    sev_prec = tp / max(int(pred_high.sum()), 1)
    out += ["## 3. High-severity detection", "",
            f"Recall **{sev_rec:.3f}** · precision **{sev_prec:.3f}** ({int(true_high.sum())} true high-severity reviews)", ""]

    # 4. Crisis detection vs injected event
    # Same detector inputs the app uses: out-of-fold model sentiment + extracted severity
    df["pred_sentiment"], _ = SentimentModel().fit_predict_oof(df["text_clean"], df["rating"])
    df["severity"] = [e["severity"] for e in extracted]
    _, incidents = crisis.detect(df)
    events = json.loads(EVENTS_JSON.read_text())
    c = events["crisis"]
    c_start, c_end = pd.Timestamp(c["start_date"]).date(), pd.Timestamp(c["end_date"]).date()
    hits = incidents[(incidents["location_id"] == c["location_id"]) & (incidents["end"] >= c_start) & (incidents["start"] <= c_end)]
    delay = (hits["start"].min() - c_start).days if not hits.empty else None
    false_alarms = len(incidents) - len(hits)
    loc_days = df["location_id"].nunique() * df["date"].nunique()
    out += ["## 4. Crisis early-warning", "",
            f"Injected crisis: {c['location_id']} food-safety incident {c_start} → {c_end}.", "",
            f"- Detected: **{'yes' if not hits.empty else 'no'}**, delay **{delay} day(s)** after onset",
            f"- Other alerts (false alarms): **{false_alarms}** over {loc_days} outlet-days "
            f"({false_alarms / loc_days:.2%} of outlet-days)", "",
            "```", incidents.to_string(index=False), "```", ""]

    # 5. Reply guardrails
    brand = load_brand()
    sample = df.sample(200, random_state=1).copy()
    sample["aspects"] = [absa.extract_offline(t)["aspects"] for t in sample["text_clean"]]
    replies = [generate_reply(r, brand) for r in sample.to_dict("records")]
    pass_rate = sum(x["passed"] for x in replies) / len(replies)
    human = sum(x["route"] == "human_review" for x in replies) / len(replies)
    review = sample.iloc[0].to_dict()
    adversarial = [
        "We are so sorry, we will give you a full refund and a free meal next time.",
        "Please call our manager on 9876543210.",
        "Since 2019 we have served 50000 happy guests.",
        "This was entirely our fault and we accept liability.",
        "",
    ]
    caught = sum(not guardrail_check(a, review, brand)["passed"] for a in adversarial)
    out += ["## 5. Reply drafting + guardrails", "",
            f"- Template drafts passing all guardrails: **{pass_rate:.1%}** (200 reviews)",
            f"- Routed to human approval (≤2★ or high severity): **{human:.1%}**",
            f"- Adversarial unsafe replies caught by guardrails: **{caught}/{len(adversarial)}**", ""]

    text = "\n".join(out)
    (ROOT / "eval" / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0, help="number of reviews to score with Claude (needs API key)")
    main(ap.parse_args().llm)
