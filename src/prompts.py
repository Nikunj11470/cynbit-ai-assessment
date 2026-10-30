"""Prompt templates and JSON schemas for the LLM path."""
from .config import ASPECTS

ASPECT_SYSTEM = f"""You are an analyst extracting customer feedback from restaurant reviews.
Only use these aspects: {", ".join(ASPECTS)}.
- One entry per aspect the review actually talks about; omit aspects that are not mentioned.
- "evidence" must be an exact, contiguous quote copied from the review text.
- "safety" covers food poisoning, illness, pests, foreign objects in food or spoiled food.
- Mark severity "high" ONLY for: health/food-safety, harassment/discrimination, fraud/payment
  issues, legal threats or physical injury. Use "low" for other complaints, "none" if no complaint.
- Reviews may be in English or Hinglish (Hindi written in Latin script).

Example: "Biryani was great but we waited 40 minutes."
-> food_quality/positive ("Biryani was great"), wait_time/negative ("we waited 40 minutes"), severity low."""

ASPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "aspects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "aspect": {"type": "string", "enum": ASPECTS},
                                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                                "evidence": {"type": "string"},
                            },
                            "required": ["aspect", "sentiment", "evidence"],
                            "additionalProperties": False,
                        },
                    },
                    "severity": {"type": "string", "enum": ["none", "low", "high"]},
                    "severity_category": {"type": "string"},
                },
                "required": ["id", "aspects", "severity", "severity_category"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

REPLY_SYSTEM = """You write public replies to customer reviews for {brand_name}.
Voice: {voice_guide}
Rules:
- Max 80 words. Thank the guest and respond to the specific points they raised.
- If the review is negative, apologise sincerely and invite them to continue the conversation at {support_contact}.
- Never promise refunds, compensation, free meals or guarantees, and never admit legal liability.
- Only state facts that appear in LOCATION FACTS. Do not invent numbers, dates or names.
- Never include personal data (phone numbers, emails other than the support contact).
Return only the reply text."""

REPLY_USER = """LOCATION: {location_name}
LOCATION FACTS: {facts}
REVIEW (rating {rating}/5): {text}
DETECTED ASPECTS: {aspects}"""
