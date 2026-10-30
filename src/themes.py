"""Complaint theme discovery: cluster negative evidence snippets into recurring problems.

PoC: TF-IDF -> LSA (truncated SVD) -> K-Means, k chosen by silhouette score (lightweight, CPU-only).
Production upgrade: sentence-transformer embeddings + UMAP + HDBSCAN (BERTopic) and LLM-written labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Dish names would otherwise split one problem ("X was cold") into one cluster per dish
DOMAIN_STOP = {"paneer", "tikka", "dal", "makhani", "butter", "chicken", "biryani", "masala", "dosa",
               "chole", "bhature", "laal", "maas", "thali", "tadka", "lane", "really", "way",
               # Hinglish function words
               "tha", "thi", "hai", "bahut", "aur", "ka", "ki", "ke", "mein", "bilkul"}


def find_themes(aspects_long: pd.DataFrame, k_range=range(8, 25, 2), seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (themes summary, negative snippets with theme_id)."""
    neg = aspects_long[aspects_long["sentiment"] == "negative"].copy()
    if len(neg) < 20:
        return pd.DataFrame(), neg.assign(theme_id=-1)

    vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP), ngram_range=(1, 2), min_df=3)
    X_tfidf = vec.fit_transform(neg["evidence"].str.lower())
    # LSA: dense semantic space so short snippets with no shared words aren't dumped into one cluster
    svd = TruncatedSVD(n_components=min(60, X_tfidf.shape[1] - 1), random_state=seed)
    X = normalize(svd.fit_transform(X_tfidf))

    best = None
    for k in k_range:
        if k >= X.shape[0]:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        score = silhouette_score(X, km.labels_, metric="cosine", sample_size=min(2000, X.shape[0]), random_state=seed)
        if best is None or score > best[0]:
            best = (score, km)
    km = best[1]
    neg["theme_id"] = km.labels_

    terms = np.array(vec.get_feature_names_out())
    rows = []
    for tid, grp in neg.groupby("theme_id"):
        centroid = np.asarray(X_tfidf[(km.labels_ == tid)].mean(axis=0)).ravel()
        top_terms = terms[np.argsort(centroid)[::-1][:4]]
        rows.append({
            "theme_id": tid,
            "representative": grp["evidence"].str.lower().str.strip(" .").mode().iat[0],
            "label": " / ".join(top_terms),
            "mentions": len(grp),
            "share": len(grp) / len(neg),
            "main_aspect": grp["aspect"].mode().iat[0],
            "top_location": grp["location_name"].mode().iat[0],
            "examples": grp["evidence"].drop_duplicates().head(3).tolist(),
        })
    themes = pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)
    themes.attrs["silhouette"] = round(float(best[0]), 3)
    return themes, neg
