"""Grounded, brand-safe reply drafting with guardrails and human-in-the-loop routing."""
from __future__ import annotations

import json
import re

from . import llm
from .config import ASPECT_LABELS, BRAND_PROFILE
from .ingest import EMAIL_RE, PHONE_RE
from .prompts import REPLY_SYSTEM, REPLY_USER

FORBIDDEN_RE = re.compile(r"\brefund|\bcompensat|\bliab(?:le|ility)\b|\bour fault\b|\bguarantee|\bfree (?:meal|dinner|food)\b"
                          r"|\bvoucher\b|\bwe admit\b", re.IGNORECASE)
MAX_WORDS = 90

ACK = {
    ("food_quality", "positive"): "we're delighted you enjoyed the food",
    ("food_quality", "negative"): "we're sorry the food did not meet the standard you expect from us",
    ("service", "positive"): "it's great to hear the service was smooth",
    ("service", "negative"): "we apologise for the slow and inconsistent service",
    ("staff", "positive"): "we'll pass your kind words on to the team",
    ("staff", "negative"): "we're sorry our team was not as courteous as they should have been",
    ("wait_time", "positive"): "we're glad you didn't have to wait",
    ("wait_time", "negative"): "we're sorry for the long wait",
    ("price", "positive"): "we're happy you found it good value",
    ("price", "negative"): "we hear your concern about pricing and value",
    ("cleanliness", "positive"): "thank you for noticing the cleanliness",
    ("cleanliness", "negative"): "we're sorry the outlet was not as clean as it should be",
    ("delivery", "positive"): "we're glad your delivery arrived well",
    ("delivery", "negative"): "we're sorry your delivery experience fell short",
    ("ambience", "positive"): "we're glad you liked the ambience",
    ("ambience", "negative"): "thank you for flagging the issues with the seating and comfort",
    ("app_website", "positive"): "we're glad online ordering was easy",
    ("app_website", "negative"): "we're sorry about the trouble with our website",
    ("safety", "negative"): "we take what you've described extremely seriously and our kitchen team is investigating",
}


def load_brand() -> dict:
    return json.loads(BRAND_PROFILE.read_text())


def _location_facts(brand: dict, location_id: str) -> list[str]:
    loc = brand["locations"].get(location_id, {})
    return [f"Opening hours: {loc['hours']}"] + loc.get("facts", []) if loc else []


def needs_human(review: dict) -> list[str]:
    reasons = []
    if review.get("severity") == "high":
        reasons.append(f"high-severity issue ({review.get('severity_category')})")
    if review.get("rating", 5) <= 2:
        reasons.append("rating <= 2")
    return reasons


def guardrail_check(reply: str, review: dict, brand: dict) -> dict:
    facts = " ".join(_location_facts(brand, review["location_id"]))
    allowed_numbers = set(re.findall(r"\d+", review["text"] + " " + facts))
    emails = {e for e in EMAIL_RE.findall(reply) if e != brand["support_contact"]}

    checks = {
        "No refunds / compensation / liability promises": not FORBIDDEN_RE.search(reply),
        "No personal data": not (PHONE_RE.search(reply) or emails),
        f"Length <= {MAX_WORDS} words": len(reply.split()) <= MAX_WORDS,
        "No ungrounded numbers": set(re.findall(r"\d+", reply)) <= allowed_numbers,
        "Not empty": bool(reply.strip()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    escalate = needs_human(review)
    return {
        "checks": checks,
        "passed": not failed,
        "route": "human_review" if (failed or escalate) else "auto_post",
        "reasons": failed + escalate,
    }


def draft_offline(review: dict, brand: dict) -> str:
    aspects = review.get("aspects") or []
    negative = [a for a in aspects if a["sentiment"] == "negative"]
    opening = "Thank you for taking the time to share this" if negative else "Thank you so much for your lovely review"

    acks = []
    for a in sorted(aspects, key=lambda a: a["sentiment"] != "negative")[:2]:
        text = ACK.get((a["aspect"], a["sentiment"]))
        if text and text not in acks:
            acks.append(text)
    body = ". " + (", and ".join(acks)).capitalize() + "." if acks else "."

    extra = ""
    if negative:
        facts = brand["locations"].get(review["location_id"], {}).get("facts", [])
        neg_aspects = {a["aspect"] for a in negative}
        relevant = [f for f in facts
                    if ("delivery" in neg_aspects and ("deliver" in f.lower() or "packaging" in f.lower()))
                    or ("cleanliness" in neg_aspects and "clean" in f.lower())
                    or ("wait_time" in neg_aspects and "staff" in f.lower())
                    or ("safety" in neg_aspects and "safety" in f.lower())]
        if relevant:
            extra += f" For context: {relevant[0][0].lower() + relevant[0][1:]}"
        extra += (f" We have shared your feedback with our {review['location_name']} team. "
                  f"Please write to us at {brand['support_contact']} so we can make this right.")
    else:
        extra = " We look forward to welcoming you back soon!"
    return f"{opening}{body}{extra}\n- Team {brand['brand_name']}"


def draft_llm(review: dict, brand: dict) -> str:
    system = REPLY_SYSTEM.format(**brand)
    user = REPLY_USER.format(
        location_name=review["location_name"],
        facts="; ".join(_location_facts(brand, review["location_id"])),
        rating=review["rating"], text=review["text_clean"],
        aspects=", ".join(f"{ASPECT_LABELS[a['aspect']]} ({a['sentiment']})" for a in review.get("aspects", [])) or "none",
    )
    return llm.complete(system, user, max_tokens=4000)


def generate_reply(review: dict, brand: dict, use_llm: bool = False) -> dict:
    """Draft -> guardrails -> (one regeneration if the LLM draft fails) -> route."""
    source = "offline-template"
    reply = draft_offline(review, brand)
    if use_llm:
        try:
            reply, source = draft_llm(review, brand), "claude"
            if not guardrail_check(reply, review, brand)["passed"]:
                reply, source = draft_llm(review, brand), "claude (regenerated)"
        except llm.LLMError as e:
            source = f"offline-template (LLM unavailable: {e})"
    return {"reply": reply, "source": source, **guardrail_check(reply, review, brand)}
