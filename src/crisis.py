"""Crisis early-warning per location.

Two independent triggers (either raises an alert):
  1. Statistical spike - 3-day rolling negative-review share compared with that location's own
     previous 28-day baseline using a binomial z-test (accounts for how many reviews the window has),
     plus a minimum absolute uplift. Interpretable, needs no labelled data, tunable per brand.
  2. Severity override - >= N high-severity reviews (food safety, harassment, fraud, legal) in 3 days.
Consecutive alert days are merged into one incident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 3
BASELINE_DAYS = 28


def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(neg=df["pred_sentiment"].eq("negative"), high=df["severity"].eq("high"))
    daily = g.groupby(["location_id", "date"]).agg(
        n=("review_id", "size"), neg=("neg", "sum"), high=("high", "sum"), rating=("rating", "mean"))
    full_idx = pd.MultiIndex.from_product(
        [daily.index.levels[0], pd.date_range(df["date"].min(), df["date"].max(), freq="D")],
        names=["location_id", "date"])
    return daily.reindex(full_idx).fillna({"n": 0, "neg": 0, "high": 0}).reset_index()


def detect(df: pd.DataFrame, z_threshold: float = 3.5, min_reviews: int = 10, min_uplift: float = 0.2,
           high_threshold: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (daily signal table, incidents table)."""
    daily = daily_stats(df)
    frames = []
    for loc, d in daily.groupby("location_id"):
        d = d.sort_values("date").copy()
        roll_n = d["n"].rolling(WINDOW, min_periods=1).sum()
        roll_neg = d["neg"].rolling(WINDOW, min_periods=1).sum()
        d["roll_n"] = roll_n
        d["neg_share"] = np.where(roll_n > 0, roll_neg / roll_n.replace(0, np.nan), np.nan)
        # Baseline = negative share over the previous 28 days (excluding the current window)
        base_neg = d["neg"].shift(WINDOW).rolling(BASELINE_DAYS, min_periods=14).sum()
        base_n = d["n"].shift(WINDOW).rolling(BASELINE_DAYS, min_periods=14).sum()
        d["baseline"] = (base_neg / base_n.replace(0, np.nan)).clip(0.02, 0.98)
        # Binomial z-test: is this window's negative share higher than chance given its review count?
        se = np.sqrt(d["baseline"] * (1 - d["baseline"]) / roll_n.replace(0, np.nan))
        d["z"] = (d["neg_share"] - d["baseline"]) / se
        d["high_3d"] = d["high"].rolling(WINDOW, min_periods=1).sum()
        d["spike"] = ((d["z"] >= z_threshold) & (d["roll_n"] >= min_reviews)
                      & (d["neg_share"] - d["baseline"] >= min_uplift))
        d["severity_alert"] = d["high_3d"] >= high_threshold
        d["alert"] = d["spike"] | d["severity_alert"]
        frames.append(d)
    signal = pd.concat(frames, ignore_index=True)

    incidents = []
    for loc, d in signal[signal["alert"]].groupby("location_id"):
        run_id = (d["date"].diff().dt.days != 1).cumsum()
        for _, run in d.groupby(run_id):
            reasons = []
            if run["severity_alert"].any():
                reasons.append(f"{int(run['high_3d'].max())} high-severity reviews in {WINDOW} days")
            if run["spike"].any():
                reasons.append(f"negative share spiked to {run['neg_share'].max():.0%} "
                               f"(baseline {run['baseline'].mean():.0%}, z={run['z'].max():.1f})")
            incidents.append({
                "location_id": loc,
                "start": run["date"].min().date(),
                "end": run["date"].max().date(),
                "days": len(run),
                "peak_neg_share": round(float(run["neg_share"].max()), 3),
                "high_severity_reviews": int(run["high"].sum()),
                "trigger": "; ".join(reasons),
            })
    cols = ["location_id", "start", "end", "days", "peak_neg_share", "high_severity_reviews", "trigger"]
    return signal, pd.DataFrame(incidents, columns=cols).sort_values("start", ascending=False, ignore_index=True)
