"""Generate a realistic multi-location review dataset for the PoC.

Why synthetic? Revio's real client reviews are private, and public datasets
(e.g. Yelp) have no Indian multi-location chains or labelled aspects. A
generator lets us:
  * control ground truth (aspect labels, severity) for offline evaluation,
  * inject known events (a food-safety crisis, a slow delivery decline)
    so we can measure whether the system detects them.

The brand "Tadka Lane" is fictional. Run:  python data/generate_data.py
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "sample_reviews.csv"
OUT_EVENTS = ROOT / "data" / "injected_events.json"

START = date(2026, 7, 2)
DAYS = 120

LOCATIONS = {
    "JPR-CSCHEME": {"name": "Jaipur - C-Scheme", "volume": 4.0, "bias": {}},
    "JPR-MALVIYA": {"name": "Jaipur - Malviya Nagar", "volume": 3.0, "bias": {"cleanliness": -0.6}},
    "DEL-CP": {"name": "Delhi - Connaught Place", "volume": 5.0, "bias": {}},
    "MUM-BANDRA": {"name": "Mumbai - Bandra", "volume": 4.5, "bias": {"wait_time": -1.0}},
    "BLR-INDIRA": {"name": "Bengaluru - Indiranagar", "volume": 3.5, "bias": {"food_quality": 0.7, "price": -0.9}},
    "PUN-KP": {"name": "Pune - Koregaon Park", "volume": 3.0, "bias": {}},
}

# Injected events (ground truth for the crisis detector and the rating-driver view)
CRISIS = {"location_id": "DEL-CP", "start_day": 108, "end_day": 117, "type": "health_safety"}
DRIFT = {"location_id": "PUN-KP", "start_day": 70, "aspect": "delivery"}

MENTION_PROB = {
    "food_quality": 0.55, "service": 0.30, "staff": 0.25, "wait_time": 0.25, "price": 0.20,
    "cleanliness": 0.15, "delivery": 0.20, "ambience": 0.20, "app_website": 0.07,
}
BASE_POS_LOGIT = 0.95  # ~72% of aspect mentions are positive

DISHES = ["paneer tikka", "dal makhani", "butter chicken", "biryani", "masala dosa",
          "chole bhature", "laal maas", "thali"]
NAMES = ["Rohan", "Priya", "Aman", "Neha", "Kabir"]
PLATFORMS = ["Google", "Zomato", "Facebook", "Website"]

TEMPLATES = {
    "food_quality": {
        "positive": ["the {dish} was absolutely delicious", "food was fresh and full of flavour",
                     "loved the {dish} - perfectly spiced", "best {dish} I've had in the city",
                     "portions were generous and tasty"],
        "negative": ["the {dish} was cold and bland", "food tasted stale", "the {dish} was way too oily",
                     "the {dish} was undercooked", "quality of food has really gone down"],
        "hi_positive": ["khana bahut tasty tha", "{dish} ka swad zabardast tha"],
        "hi_negative": ["khana bilkul bekaar tha", "{dish} thanda aur feeka tha"],
    },
    "service": {
        "positive": ["service was quick and attentive", "they took great care of our table",
                     "service was smooth from start to finish"],
        "negative": ["service was painfully slow and disorganised", "we had to ask three times for water",
                     "our order came out wrong twice"],
    },
    "staff": {
        "positive": ["staff were very polite and friendly", "our server {name} was super helpful",
                     "the manager personally checked on us"],
        "negative": ["staff were rude and dismissive", "the waiter ignored us for most of the evening",
                     "the staff argued with us about the bill"],
        "hi_positive": ["staff bahut friendly tha"],
        "hi_negative": ["staff ka behaviour bahut rude tha"],
    },
    "wait_time": {
        "positive": ["food arrived within 10 minutes", "no waiting even on a weekend", "we got seated right away"],
        "negative": ["we waited over 45 minutes for our food", "had to wait an hour for a table despite booking",
                     "it took forever to get the bill"],
        "hi_negative": ["khane mein bahut der lagi"],
    },
    "price": {
        "positive": ["great value for money", "prices are very reasonable for the quality",
                     "affordable for a family dinner"],
        "negative": ["way too expensive for the portion size", "overpriced for what you get",
                     "they added a hidden service charge"],
        "hi_positive": ["paisa vasool"],
        "hi_negative": ["bahut mehenga hai"],
    },
    "cleanliness": {
        "positive": ["the place was spotless", "washrooms were clean", "tables were neat and hygienic"],
        "negative": ["tables were sticky and dirty", "the washroom was filthy", "saw flies near the counter"],
    },
    "delivery": {
        "positive": ["delivery was quick and the food was still hot", "packaging was excellent and nothing spilled",
                     "delivery arrived earlier than expected"],
        "negative": ["delivery was over an hour late", "the delivery order arrived cold",
                     "items were missing from my delivery", "packaging leaked all over the bag"],
    },
    "ambience": {
        "positive": ["lovely ambience and music", "great vibe for a date night", "beautiful interiors"],
        "negative": ["it was too loud to have a conversation", "the AC wasn't working",
                     "seating was cramped and uncomfortable"],
    },
    "app_website": {
        "positive": ["online booking was super easy", "ordering through the website was smooth"],
        "negative": ["the website kept crashing during payment", "online booking didn't go through",
                     "the app showed wrong menu prices"],
    },
    "safety": {
        "negative": ["I got food poisoning after eating here", "found a hair in my {dish}",
                     "my whole family fell sick after dinner", "found a cockroach near our table",
                     "the chicken smelled spoiled"],
    },
}

OPENERS = ["", "", "Visited with family. ", "Came here for a friend's birthday. ",
           "Ordered for a work lunch. ", "Second visit. "]
CLOSERS = {
    "positive": ["", " Will come again!", " Highly recommend."],
    "neutral": ["", " Overall okay."],
    "negative": ["", " Not coming back.", " Very disappointed."],
}


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def pick_clause(rng: random.Random, aspect: str, polarity: str) -> str:
    bank = TEMPLATES[aspect]
    hi_key = f"hi_{polarity}"
    pool = bank[hi_key] if hi_key in bank and rng.random() < 0.15 else bank[polarity]
    return rng.choice(pool).format(dish=rng.choice(DISHES), name=rng.choice(NAMES))


def make_review(rng: random.Random, loc_id: str, day: int) -> dict:
    bias = LOCATIONS[loc_id]["bias"]
    in_crisis = loc_id == CRISIS["location_id"] and CRISIS["start_day"] <= day <= CRISIS["end_day"]
    in_drift = loc_id == DRIFT["location_id"] and day >= DRIFT["start_day"]

    labels: list[tuple[str, str]] = []
    for aspect, p in MENTION_PROB.items():
        if in_drift and aspect == DRIFT["aspect"]:
            p = 0.45
        if rng.random() < p:
            logit = BASE_POS_LOGIT + bias.get(aspect, 0.0)
            if in_drift and aspect == DRIFT["aspect"]:
                progress = (day - DRIFT["start_day"]) / (DAYS - DRIFT["start_day"])
                logit -= 3.2 * progress
            labels.append((aspect, "positive" if rng.random() < sigmoid(logit) else "negative"))

    # Rare background safety complaints everywhere; a burst during the crisis
    if (in_crisis and rng.random() < 0.40) or rng.random() < 0.003:
        # A guest reporting a safety issue rarely praises the meal: keep one other, negative mention
        labels = [(a, "negative") for a, _ in labels if a != "food_quality"][:1] + [("safety", "negative")]

    if not labels:
        labels.append(("food_quality", "positive" if rng.random() < 0.72 else "negative"))
    labels = labels[:4]
    rng.shuffle(labels)

    clauses = [pick_clause(rng, a, s) for a, s in labels]
    score = np.mean([1 if s == "positive" else -1 for _, s in labels])
    has_safety = any(a == "safety" for a, _ in labels)

    text = clauses[0][0].upper() + clauses[0][1:]
    joiner = ""
    for i in range(1, len(clauses)):
        clause = clauses[i]
        prev_pol, nxt_pol = labels[i - 1][1], labels[i][1]
        contrast = prev_pol != nxt_pol and joiner != " but " and rng.random() < 0.6
        joiner = " but " if contrast else rng.choice([", ", ". "])
        clause = clause[0].upper() + clause[1:] if joiner == ". " else clause
        text += joiner + clause
    text += "."

    overall = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"
    text = rng.choice(OPENERS) + text + rng.choice(CLOSERS[overall])

    stars = 1 if has_safety else int(np.clip(round(3 + 2 * score + rng.gauss(0, 0.6)), 1, 5))

    return {
        "location_id": loc_id,
        "location_name": LOCATIONS[loc_id]["name"],
        "platform": rng.choice(PLATFORMS),
        "created_at": (START + timedelta(days=day)).isoformat(),
        "rating": stars,
        "text": text,
        # ground-truth columns: used ONLY by eval/run_eval.py, never by the pipeline
        "true_aspects": json.dumps([{"aspect": a, "sentiment": s} for a, s in sorted(set(labels))]),
        "true_severity": "high" if has_safety else "none",
    }


def main() -> None:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)
    rows = []
    for day in range(DAYS):
        weekend = (START + timedelta(days=day)).weekday() >= 5
        for loc_id, meta in LOCATIONS.items():
            lam = meta["volume"] * (1.4 if weekend else 1.0)
            if loc_id == CRISIS["location_id"] and CRISIS["start_day"] <= day <= CRISIS["end_day"]:
                lam *= 1.8
            for _ in range(np_rng.poisson(lam)):
                rows.append(make_review(rng, loc_id, day))

    # A few reviews with personal data (to demo PII scrubbing)
    for i in rng.sample(range(len(rows)), 6):
        rows[i]["text"] += rng.choice([" Call me on 9876543210 to sort this out.",
                                        " Email me at rahul.sharma@example.com."])
    # Cross-posted duplicates (same review on two platforms) to demo de-duplication
    for i in rng.sample(range(len(rows)), 25):
        dup = dict(rows[i])
        dup["platform"] = "Google" if dup["platform"] != "Google" else "Facebook"
        rows.append(dup)

    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    df.insert(0, "review_id", [f"R{i:05d}" for i in range(len(df))])
    df.to_csv(OUT_CSV, index=False)

    events = {
        "crisis": {**CRISIS, "start_date": (START + timedelta(days=CRISIS["start_day"])).isoformat(),
                   "end_date": (START + timedelta(days=CRISIS["end_day"])).isoformat()},
        "drift": {**DRIFT, "start_date": (START + timedelta(days=DRIFT["start_day"])).isoformat()},
    }
    OUT_EVENTS.write_text(json.dumps(events, indent=2))
    print(f"Wrote {len(df)} reviews to {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
