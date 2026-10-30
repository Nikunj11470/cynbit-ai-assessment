"""End-to-end pipeline: ingest -> sentiment -> aspects/severity -> themes -> crisis alerts."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import aspects as absa
from . import crisis, themes
from .ingest import load_reviews
from .sentiment import SentimentModel


@dataclass
class PipelineResult:
    reviews: pd.DataFrame
    aspects_long: pd.DataFrame
    themes: pd.DataFrame
    signal: pd.DataFrame
    incidents: pd.DataFrame
    sentiment_model: SentimentModel
    stats: dict = field(default_factory=dict)


def explode_aspects(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"review_id": r.review_id, "location_id": r.location_id, "location_name": r.location_name,
         "date": r.date, "rating": r.rating, **a}
        for r in reviews.itertuples() for a in r.aspects
    ]
    return pd.DataFrame(rows, columns=["review_id", "location_id", "location_name", "date", "rating",
                                       "aspect", "sentiment", "evidence"])


def run(reviews: pd.DataFrame | None = None, use_llm: bool = False, llm_limit: int = 200) -> PipelineResult:
    if reviews is None:
        reviews, stats = load_reviews()
    else:
        stats = {}
    reviews = reviews.copy()

    model = SentimentModel()
    reviews["pred_sentiment"], reviews["sentiment_conf"] = model.fit_predict_oof(reviews["text_clean"], reviews["rating"])

    extracted = {rid: absa.extract_offline(t) for rid, t in zip(reviews["review_id"], reviews["text_clean"])}
    if use_llm:
        # LLM cost control: only the most recent reviews go to Claude in the demo
        recent = reviews.nlargest(llm_limit, "created_at")
        extracted.update(absa.extract_llm(list(zip(recent["review_id"], recent["text_clean"]))))
        stats["llm_reviews"] = len(recent)
    reviews["aspects"] = reviews["review_id"].map(lambda r: extracted[r]["aspects"])
    reviews["severity"] = reviews["review_id"].map(lambda r: extracted[r]["severity"])
    reviews["severity_category"] = reviews["review_id"].map(lambda r: extracted[r]["severity_category"])

    long = explode_aspects(reviews)
    theme_df, _ = themes.find_themes(long)
    signal, incidents = crisis.detect(reviews)
    return PipelineResult(reviews, long, theme_df, signal, incidents, model, stats)


def analyse_text(text: str, model: SentimentModel) -> dict:
    """Live analysis of a single pasted review (used by the 'Try it' tab)."""
    from .ingest import detect_language, scrub_pii
    clean = scrub_pii(text)
    label, conf = model.predict([clean])[0]
    return {"text_clean": clean, "language": detect_language(clean),
            "pred_sentiment": label, "sentiment_conf": conf, **absa.extract_offline(clean)}
