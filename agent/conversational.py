import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from agent.schema import ResponseType

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")


class ConversationalIntent(str, Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    NONE = "none"


SUBSTANTIVE_KEYWORDS = {
    "order", "orders", "nyk", "ord", "trk", "tracking", "track", "shipment",
    "courier", "package", "parcel", "delivery", "deliveries", "dispatch",
    "dispatched", "arrive", "arrival", "delayed", "delay", "transit",
    "return", "returns", "returnable", "exchange", "refund", "refunds",
    "cancel", "cancellation", "cancelling", "damaged", "broken", "defect",
    "defective", "replace", "replacement", "rma",
    "policy", "policies", "window", "sla", "warranty", "guarantee", "cod",
    "cash", "payment", "international", "loyalty", "points", "tier",
    "rewards", "reward", "balance", "terms", "conditions",
    "beauty", "cosmetic", "cosmetics", "makeup", "fragrance", "fragrances",
    "perfume", "lipstick", "lipsticks", "skincare", "item", "items", "product",
    "products",
    "what", "what's", "when", "where", "why", "how", "which", "who", "whom",
    "status", "check", "eligible", "eligibility", "can", "could", "would",
    "want", "need", "wish", "please check", "tell me", "find", "show",
}

OPERATIONAL_ID_PATTERN = re.compile(
    r"\b(nyk-\d{5}|ord\d+|trk-\d+|cust-\d{5})\b",
    re.IGNORECASE,
)

GREETING_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hey there",
    "hello there",
    "hi there",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "good day",
    "namaste",
    "morning",
    "afternoon",
    "evening",
    "what's up",
    "howdy",
}

THANKS_PHRASES = {
    "thanks",
    "thank you",
    "thanks a lot",
    "thank you so much",
    "thank you very much",
    "many thanks",
    "thx",
    "ty",
    "thanks a bunch",
    "appreciate it",
}

FILLER_WORDS = {
    "nykaa", "nykaaassist", "assist", "support", "team", "bot", "there",
    "all", "everyone", "please", "you", "very", "much", "so", "a", "lot",
    "for", "the", "your", "help", "helping", "friend", "dear", "sir", "madam",
}


def get_kolkata_now() -> datetime:
    return datetime.now(KOLKATA_TZ)


def get_kolkata_daypart(target_datetime: Optional[datetime] = None) -> str:
    if target_datetime is None:
        target_datetime = get_kolkata_now()
    elif target_datetime.tzinfo is None:
        target_datetime = target_datetime.replace(tzinfo=KOLKATA_TZ)
    else:
        target_datetime = target_datetime.astimezone(KOLKATA_TZ)

    time_tuple = (target_datetime.hour, target_datetime.minute)
    if (5, 0) <= time_tuple <= (11, 59):
        return "morning"
    elif (12, 0) <= time_tuple <= (16, 59):
        return "afternoon"
    else:
        return "evening"


def detect_conversational_intent(query_text: str) -> ConversationalIntent:
    if not query_text or not query_text.strip():
        return ConversationalIntent.NONE

    cleaned = re.sub(r"[^\w\s]", " ", query_text.lower()).strip()
    words = cleaned.split()
    if not words:
        return ConversationalIntent.NONE

    if OPERATIONAL_ID_PATTERN.search(query_text):
        return ConversationalIntent.NONE

    if any(word in SUBSTANTIVE_KEYWORDS for word in words):
        return ConversationalIntent.NONE

    joined = " ".join(words)

    for phrase in sorted(THANKS_PHRASES, key=len, reverse=True):
        if phrase in joined:
            residual = set(joined.replace(phrase, " ").split()) - FILLER_WORDS
            if not residual:
                return ConversationalIntent.THANKS

    for phrase in sorted(GREETING_PHRASES, key=len, reverse=True):
        if phrase in joined:
            residual = set(joined.replace(phrase, " ").split()) - FILLER_WORDS
            if not residual:
                return ConversationalIntent.GREETING

    return ConversationalIntent.NONE


def generate_conversational_response(
    intent: ConversationalIntent,
    target_datetime: Optional[datetime] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    active_trace_id = trace_id or str(uuid.uuid4())

    if intent == ConversationalIntent.THANKS:
        answer = "You're welcome! 😊 Let me know if you need any more help."
    else:
        daypart = get_kolkata_daypart(target_datetime)
        if daypart == "morning":
            answer = (
                "Good morning! 👋 How can I help you with your Nykaa order, "
                "delivery, returns, or other support needs?"
            )
        elif daypart == "afternoon":
            answer = (
                "Good afternoon! 👋 How can I help you with your Nykaa support needs?"
            )
        else:
            answer = (
                "Good evening! 👋 How can I help you today?"
            )

    return {
        "response_type": ResponseType.CONVERSATIONAL.value,
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
        "escalation_score": None,
        "trace_id": active_trace_id,
    }
