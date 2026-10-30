"""Aspect-Based Sentiment Analysis (ABSA) + severity tagging.

Two interchangeable extractors with the same output format:
  * `extract_offline` - fast, free, deterministic lexicon + clause rules (default)
  * `extract_llm`     - Claude with a strict JSON schema (higher recall on unseen phrasing)

Output per review: {"aspects": [{"aspect", "sentiment", "evidence"}], "severity", "severity_category"}
"""
from __future__ import annotations

import json
import re

from . import llm
from .prompts import ASPECT_SCHEMA, ASPECT_SYSTEM

DISH_RE = r"\b(?:paneer|dal|chicken|biryani|dosa|chole|bhature|laal maas|thali|curry|naan|roti)\b"

ASPECT_PATTERNS = {
    "food_quality": [r"\bfood\b", r"\bdish", DISH_RE, r"\bkhana\b", r"\bmeal\b", r"\bportions?\b"],
    "service": [r"\bservice\b(?! charge)", r"\btook great care\b", r"\bask(?:ed)? \w+ times\b", r"\border came out\b"],
    "staff": [r"\bstaff\b", r"\bwaiter", r"\bserver\b", r"\bmanager\b", r"\bbehaviou?r\b"],
    "wait_time": [r"\bwait(?:ed|ing)?\b", r"\bminutes\b", r"\bseated\b", r"\bforever\b", r"\bder lagi\b",
                  r"\bright away\b", r"\bfor a table\b"],
    "price": [r"\bpric(?:e|es|ey|ed)\b", r"\bexpensive\b", r"\boverpriced\b", r"\bvalue for money\b",
              r"\baffordable\b", r"\bservice charge\b", r"\bmehenga\b", r"\bpaisa vasool\b"],
    "cleanliness": [r"\bspotless\b", r"\bclean", r"\bwashrooms?\b", r"\bhygien", r"\bsticky\b", r"\bdirty\b",
                    r"\bfilthy\b", r"\bflies\b"],
    "delivery": [r"\bdeliver", r"\bpackaging\b", r"\bspilled\b", r"\bmissing from\b", r"\bleaked\b"],
    "ambience": [r"\bambience\b", r"\bmusic\b", r"\bvibe\b", r"\binteriors?\b", r"\btoo loud\b", r"\bthe ac\b",
                 r"\bseating\b", r"\bcramped\b"],
    "app_website": [r"\bwebsite\b", r"\bapp\b", r"\bonline\b"],
    "safety": [r"\bpoisoning\b", r"\bfell sick\b", r"\bsick after\b", r"\bcockroach", r"\bhair in\b",
               r"\bspoiled\b", r"\bsmelled\b", r"\brat\b", r"\binsect"],
}
# Words that make a clause unambiguously about food quality (vs. a generic "food" mention)
STRONG_FOOD_RE = re.compile(r"tast|flavou?r|bland|stale|oily|undercooked|delicious|spiced|swad|feeka|"
                            r"quality of food|bekaar")

POSITIVE = ["delicious", "fresh", "loved", "best", "generous", "tasty", "perfectly", "quick", "attentive",
            "great care", "smooth", "polite", "friendly", "helpful", "personally checked", "within",
            "no waiting", "right away", "great value", "value for money", "reasonable", "affordable", "spotless",
            "clean", "neat", "hygienic", "still hot", "excellent", "nothing spilled", "earlier than expected",
            "lovely", "great vibe", "beautiful", "super easy", "zabardast", "vasool", "recommend", "amazing"]
NEGATIVE = ["cold", "bland", "stale", "oily", "undercooked", "gone down", "slow", "disorganised", "three times",
            "wrong", "rude", "dismissive", "ignored", "argued", "waited", "an hour", "forever", "expensive",
            "overpriced", "hidden", "sticky", "dirty", "filthy", "flies", "late", "missing", "leaked", "too loud",
            "wasn't working", "cramped", "uncomfortable", "crashing", "didn't go through", "bekaar", "thanda",
            "feeka", "der lagi", "mehenga", "poisoning", "sick", "cockroach", "spoiled", "hair", "terrible",
            "worst", "awful"]

SEVERITY_RULES = {
    "health_safety": [r"poisoning", r"fell sick", r"sick after", r"cockroach", r"hair in", r"spoiled",
                      r"\brat\b", r"insect", r"vomit", r"hospital"],
    "harassment": [r"harass", r"discriminat", r"abus(?:e|ive)", r"threaten"],
    "fraud_payment": [r"charged twice", r"fraud", r"scam", r"stole"],
    "legal": [r"consumer court", r"legal action", r"lawyer", r"police"],
}

CLAUSE_SPLIT_RE = re.compile(r"[.!?;]+\s*|,\s+|\s+but\s+", re.IGNORECASE)


def _polarity(clause: str) -> str:
    c = clause.lower()
    score = sum(w in c for w in POSITIVE) - sum(w in c for w in NEGATIVE)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def _clause_aspects(clause: str) -> set[str]:
    c = clause.lower()
    found = {a for a, pats in ASPECT_PATTERNS.items() if any(re.search(p, c) for p in pats)}
    # A generic food mention inside a delivery/wait/safety clause is not a food-quality opinion
    if "food_quality" in found and len(found) > 1 and not STRONG_FOOD_RE.search(c):
        found.discard("food_quality")
    if "safety" in found:
        found.discard("food_quality")
    return found


def severity_of(text: str, aspects: list[dict]) -> tuple[str, str]:
    t = text.lower()
    for category, pats in SEVERITY_RULES.items():
        if any(re.search(p, t) for p in pats):
            return "high", category
    if any(a["sentiment"] == "negative" for a in aspects):
        return "low", "complaint"
    return "none", "none"


def extract_offline(text: str) -> dict:
    aspects, seen = [], set()
    for clause in filter(None, (c.strip() for c in CLAUSE_SPLIT_RE.split(text))):
        found = _clause_aspects(clause)
        if not found:
            continue
        pol = _polarity(clause)
        for aspect in found:
            if aspect == "safety":
                pol = "negative"
            if pol == "neutral" or (aspect, pol) in seen:
                continue
            seen.add((aspect, pol))
            aspects.append({"aspect": aspect, "sentiment": pol, "evidence": clause})
    severity, category = severity_of(text, aspects)
    return {"aspects": aspects, "severity": severity, "severity_category": category}


def extract_llm(items: list[tuple[str, str]], batch_size: int = 15) -> dict[str, dict]:
    """items: [(review_id, text)]. Falls back to the offline extractor for any review the LLM fails on."""
    out: dict[str, dict] = {}
    texts = dict(items)
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        payload = json.dumps([{"id": rid, "text": t} for rid, t in batch], ensure_ascii=False)
        try:
            data = llm.complete(ASPECT_SYSTEM, f"Reviews:\n{payload}", schema=ASPECT_SCHEMA)
            results = data["results"]
        except (llm.LLMError, KeyError):
            results = []
        for r in results:
            rid = r.get("id")
            if rid not in texts:
                continue
            src = texts[rid].lower()
            # Hallucination guard: keep only aspects whose evidence is a real quote from the review
            valid = [a for a in r["aspects"] if a["evidence"].lower().strip(" .") in src]
            out[rid] = {"aspects": valid, "severity": r["severity"],
                        "severity_category": r["severity_category"] or "none",
                        "evidence_dropped": len(r["aspects"]) - len(valid)}
    for rid, text in items:
        out.setdefault(rid, {**extract_offline(text), "evidence_dropped": 0, "fallback": True})
    return out
