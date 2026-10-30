"""Shared paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REVIEWS_CSV = DATA_DIR / "sample_reviews.csv"
EVENTS_JSON = DATA_DIR / "injected_events.json"
BRAND_PROFILE = DATA_DIR / "brand_profile.json"

# Fixed aspect taxonomy -> stable analytics across time and clients
ASPECTS = ["food_quality", "service", "staff", "wait_time", "price",
           "cleanliness", "delivery", "ambience", "app_website", "safety"]

ASPECT_LABELS = {
    "food_quality": "Food quality", "service": "Service", "staff": "Staff",
    "wait_time": "Wait time", "price": "Price / value", "cleanliness": "Cleanliness",
    "delivery": "Delivery", "ambience": "Ambience", "app_website": "App / website",
    "safety": "Health & safety",
}

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
