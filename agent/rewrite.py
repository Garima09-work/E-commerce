import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ORDER_ID_PATTERN = re.compile(r"\bNYK-\d{5}\b", re.IGNORECASE)

ORDER_KEYWORDS = {
    "order",
    "tracking",
    "track",
    "package",
    "shipment",
    "dispatch",
    "delivery",
    "status",
    "arrive",
    "arrived",
    "shipping",
}

PRONOUN_PATTERN = re.compile(r"\b(it|they|them|this|these|those)\b", re.IGNORECASE)

SELF_CONTAINED_POLICIES = [
    ("return policy", "Nykaa's return policy"),
    ("cancellation policy", "Nykaa's order cancellation policy"),
    ("cancellation", "Nykaa's cancellation policy"),
    ("damaged item", "Nykaa's damaged item claims policy"),
    ("damaged", "Nykaa's damaged item policy"),
    ("reverse pickup", "Nykaa's reverse pickup policy"),
    ("pickup", "Nykaa's reverse pickup policy"),
    ("size exchange", "Nykaa's size exchange policy"),
    ("exchange", "Nykaa's size exchange policy"),
    ("warranty terms", "Nykaa's appliance warranty terms"),
    ("warranty", "Nykaa's warranty policy"),
    ("delivery sla", "Nykaa's delivery timelines and SLAs"),
    ("delivery timelines", "Nykaa's delivery timelines"),
    ("delivery", "Nykaa's delivery timeline policy"),
    ("refund timelines", "Nykaa's refund timelines"),
    ("refund", "Nykaa's refund timelines and policy"),
    ("loyalty points", "Nykaa's reward and loyalty points policy"),
    ("reward points", "Nykaa's reward points policy"),
    ("international shipping", "Nykaa's international shipping policy"),
    ("escalation matrix", "Nykaa's customer support escalation matrix"),
    ("escalation", "Nykaa's customer support escalation matrix"),
    ("payment failure", "Nykaa's failed payment and retry policy"),
]

PRODUCT_NOUNS = [
    "shoes",
    "footwear",
    "dress",
    "lipstick",
    "perfume",
    "fragrance",
    "foundation",
    "cream",
    "shampoo",
    "serum",
    "appliance",
    "hair dryer",
    "straightener",
    "clothing",
    "apparel",
    "beauty item",
    "beauty products",
    "skincare",
    "makeup",
    "item",
    "product",
]


def extract_policy_subject(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    for noun in PRODUCT_NOUNS:
        if re.search(rf"\b{re.escape(noun)}\b", lowered):
            return noun
    words = re.findall(r"\b[a-zA-Z]{3,}\b", lowered)
    stop_words = {
        "can",
        "what",
        "where",
        "when",
        "how",
        "why",
        "the",
        "this",
        "that",
        "under",
        "with",
        "from",
        "about",
        "policy",
        "return",
        "refund",
        "exchange",
        "nykaa",
        "product",
        "items",
    }
    candidates = [w for w in words if w not in stop_words]
    return candidates[0] if candidates else None


def resolve_order_pronouns(query: str, last_order_id: Optional[str]) -> Optional[str]:
    if not last_order_id or not ORDER_ID_PATTERN.match(last_order_id.strip()):
        return None

    norm_id = last_order_id.strip().upper()
    trimmed = query.strip()
    lowered = trimmed.lower().rstrip("?.!")

    # If the query already explicitly mentions an order ID, don't rewrite it with last_order_id
    if ORDER_ID_PATTERN.search(trimmed):
        return None

    # Check for "this order", "that order", "the order"
    if re.search(r"\b(this|that|the)\s+order\b", lowered):
        rewritten = re.sub(r"\b(this|that|the)\s+order\b", f"order {norm_id}", trimmed, flags=re.IGNORECASE)
        return rewritten

    # Check for "its order value"
    if re.search(r"\bits\s+order\s+value\b", lowered):
        rewritten = re.sub(r"\bits\s+order\s+value\b", f"the order value of order {norm_id}", trimmed, flags=re.IGNORECASE)
        return rewritten

    # Check for "its status"
    if re.search(r"\bits\s+status\b", lowered):
        rewritten = re.sub(r"\bits\s+status\b", f"the status of order {norm_id}", trimmed, flags=re.IGNORECASE)
        return rewritten

    # Check for "order value" without order ID
    if re.search(r"\b(order\s+value|value\s+of\s+(?:the\s+)?order)\b", lowered):
        rewritten = re.sub(r"\b(the\s+)?order\s+value\b", f"order value of order {norm_id}", trimmed, flags=re.IGNORECASE)
        if norm_id not in rewritten:
            rewritten = f"{trimmed} for order {norm_id}"
        return rewritten

    # Order follow-up attributes with "it" or "its"
    order_attr_keywords = (
        "delayed",
        "delay",
        "shipped",
        "dispatch",
        "dispatched",
        "delivered",
        "delivery",
        "arrive",
        "arrival",
        "status",
        "where",
        "when",
        "tracking",
        "track",
    )
    has_order_attr = any(re.search(rf"\b{kw}\b", lowered) for kw in order_attr_keywords)

    if has_order_attr:
        if re.search(r"\bits\b", lowered):
            rewritten = re.sub(r"\bits\b", f"order {norm_id}'s", trimmed, flags=re.IGNORECASE)
            return rewritten
        if re.search(r"\bit\b", lowered):
            rewritten = re.sub(r"\bit\b", f"order {norm_id}", trimmed, flags=re.IGNORECASE)
            return rewritten

    return None


def is_order_query(query: str, last_order_id: Optional[str] = None) -> bool:
    if ORDER_ID_PATTERN.search(query):
        return True
    lowered = query.lower()
    words = set(re.findall(r"\b\w+\b", lowered))
    if last_order_id and bool(words & ORDER_KEYWORDS):
        return True
    return False


def is_self_contained_query(query: str) -> bool:
    lowered = query.strip().lower()
    has_pronoun = bool(PRONOUN_PATTERN.search(lowered))
    word_count = len(lowered.split())
    if "this" in lowered and word_count <= 5:
        return False
    if has_pronoun and word_count <= 6:
        return False
    for phrase, _ in SELF_CONTAINED_POLICIES:
        if phrase in lowered:
            return True
    return word_count >= 8 and not has_pronoun


GREETING_PREFIX_PATTERN = re.compile(
    r"^(?:hi|hello|hey|greetings|namaste|good\s+(?:morning|afternoon|evening))\s*[,!:\s]+\s*",
    re.IGNORECASE,
)


def rewrite_query(
    query: str,
    last_policy_topic: Optional[str] = None,
    last_order_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    if not query or not query.strip():
        return ""

    trimmed = query.strip()

    # Strip conversational prefix if substantive query follows (e.g. "Hi, what is the return policy?")
    substantive = GREETING_PREFIX_PATTERN.sub("", trimmed)
    if substantive and len(substantive.split()) >= 2:
        trimmed = substantive[0].upper() + substantive[1:] if len(substantive) > 1 else substantive.upper()

    # Normalize brand references in general policy questions
    if re.search(r"\bnykaa(?:'s)?\b", trimmed, re.IGNORECASE):
        norm_brand = re.sub(r"\bnykaa(?:'s)?\s*", "", trimmed, flags=re.IGNORECASE).strip()
        if re.search(r"\bpolicy\b", trimmed, re.IGNORECASE) and norm_brand:
            if not norm_brand.lower().startswith(("what is the", "what are the")):
                if norm_brand.lower().startswith("what is"):
                    norm_brand = norm_brand[:7] + " the" + norm_brand[7:]
                elif norm_brand.lower().startswith("what are"):
                    norm_brand = norm_brand[:8] + " the" + norm_brand[8:]
            trimmed = norm_brand[0].upper() + norm_brand[1:] if len(norm_brand) > 1 else norm_brand.upper()

    if last_order_id:
        order_resolved = resolve_order_pronouns(trimmed, last_order_id)
        if order_resolved:
            return order_resolved

    if is_order_query(trimmed, last_order_id):
        return trimmed

    lowered = trimmed.lower().rstrip("?.!")
    subject = extract_policy_subject(last_policy_topic)

    if re.fullmatch(r"can i return this(\s+product)?", lowered):
        if subject and subject not in ("product", "item"):
            return f"Can I return my {subject} under Nykaa's return policy?"
        return "Can the previously discussed product be returned under Nykaa's return policy?"

    if "damaged" in lowered or "broken" in lowered or "defect" in lowered:
        if subject and subject not in ("product", "item"):
            return f"Can I return my damaged {subject} under Nykaa's damaged-item policy?"
        if last_policy_topic:
            return "Can I return damaged items under Nykaa's damaged-item policy?"
        return "What is Nykaa's damaged item claims policy?"

    if any(w in lowered for w in ("days", "window", "period", "timeline", "time")) and any(w in lowered for w in ("how many", "what is", "how long")):
        if subject and subject not in ("product", "item"):
            return f"Can I return my {subject}? How many days do I have to return them under the return window policy?"
        if last_policy_topic:
            clean_prev = last_policy_topic.strip().rstrip("?.!")
            return f"{clean_prev}? How many days do I have to return it under the return window policy?"
        return "What is the return window under the return window policy?"

    if "exchange" in lowered or "size" in lowered:
        if subject and subject not in ("product", "item"):
            return f"Can I exchange my {subject} for another size under Nykaa's size exchange policy?"
        return "What is Nykaa's policy on size exchanges?"

    if "pickup" in lowered or "pick up" in lowered:
        if subject and subject not in ("product", "item"):
            return f"How does reverse pickup scheduling work for {subject} under Nykaa's return policy?"
        return "How does reverse pickup scheduling work under Nykaa's return policy?"

    if "cancel" in lowered or "cancellation" in lowered:
        return "Can an order be cancelled before dispatch under Nykaa's cancellation policy?"

    if is_self_contained_query(trimmed):
        return trimmed

    if PRONOUN_PATTERN.search(lowered) and subject:
        resolved = PRONOUN_PATTERN.sub(f"the {subject}", trimmed)
        return resolved

    if last_policy_topic and len(trimmed.split()) <= 6:
        clean_topic = last_policy_topic.strip().rstrip("?.!")
        return f"{clean_topic} - {trimmed}"

    return trimmed
