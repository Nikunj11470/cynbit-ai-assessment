"""Overall review sentiment.

Model: TF-IDF (word 1-2 grams + char 3-5 grams, handles Hinglish/typos) + Logistic Regression,
trained with weak labels derived from star ratings (1-2 = negative, 3 = neutral, 4-5 = positive).
Baseline for comparison: the lexicon polarity used by the offline aspect extractor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from .aspects import NEGATIVE, POSITIVE

LABELS = ["negative", "neutral", "positive"]


def stars_to_label(stars: pd.Series) -> pd.Series:
    return pd.cut(stars, bins=[0, 2, 3, 5], labels=LABELS).astype(str)


def build_model() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def lexicon_sentiment(text: str) -> str:
    t = text.lower()
    score = sum(w in t for w in POSITIVE) - sum(w in t for w in NEGATIVE)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


class SentimentModel:
    def __init__(self) -> None:
        self.model = build_model()

    def fit_predict_oof(self, texts: pd.Series, stars: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Out-of-fold predictions (every review is scored by a model that never saw it), then fit on all."""
        y = stars_to_label(stars)
        proba = cross_val_predict(build_model(), texts, y, cv=5, method="predict_proba")
        self.model.fit(texts, y)
        classes = self.model.classes_
        return classes[proba.argmax(1)], proba.max(1)

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        proba = self.model.predict_proba(texts)
        classes = self.model.classes_
        return [(classes[p.argmax()], float(p.max())) for p in proba]
