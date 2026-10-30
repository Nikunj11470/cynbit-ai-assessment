"""Ingestion: load, normalise, de-duplicate, scrub PII, detect language."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import REVIEWS_CSV

PHONE_RE = re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
HINGLISH_WORDS = {"khana", "bahut", "tha", "hai", "bekaar", "swad", "mehenga", "der", "thanda",
                  "feeka", "zabardast", "vasool", "bilkul", "aur", "ka", "mein", "paisa", "lagi", "khane"}


def scrub_pii(text: str) -> str:
    """Mask phone numbers and emails before text is stored or sent to an LLM."""
    return EMAIL_RE.sub("[EMAIL]", PHONE_RE.sub("[PHONE]", text))


def detect_language(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    return "hinglish" if sum(t in HINGLISH_WORDS for t in tokens) >= 2 else "en"


def normalise(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    before = len(df)
    key = df["location_id"] + "|" + df["created_at"].dt.date.astype(str) + "|" + df["text"].str.lower()
    df = df.loc[~key.duplicated()]

    df["text_clean"] = df["text"].map(scrub_pii)
    df["pii_masked"] = df["text_clean"] != df["text"]
    df["language"] = df["text_clean"].map(detect_language)
    df["date"] = df["created_at"].dt.normalize()

    stats = {
        "raw_reviews": before,
        "duplicates_removed": before - len(df),
        "pii_masked": int(df["pii_masked"].sum()),
        "hinglish": int((df["language"] == "hinglish").sum()),
    }
    return df.reset_index(drop=True), stats


def load_reviews(path: Path | str = REVIEWS_CSV) -> tuple[pd.DataFrame, dict]:
    return normalise(pd.read_csv(path))
