import re
import sys
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.generate import FALLBACK_RESPONSE

SAFE_TRACE_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class ResponseType(str, Enum):
    POLICY_ANSWER = "policy_answer"
    ORDER_STATUS = "order_status"
    FALLBACK = "fallback"
    GUARDRAIL_BLOCK = "guardrail_block"
    SHIPMENT_TRACKING = "shipment_tracking"
    RETURN_STATUS = "return_status"
    RETURN_REQUEST = "return_request"
    LOYALTY_STATUS = "loyalty_status"
    CONVERSATIONAL = "conversational"
    GREETING = "greeting"
    THANKS = "thanks"


class OrderStatus(str, Enum):
    PLACED = "Placed"
    SHIPPED = "Shipped"
    DELIVERED = "Delivered"
    RETURNED = "Returned"
    REFUNDED = "Refunded"
    NOT_FOUND = "Not Found"


class OrderStatusResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^NYK-\d{5}$")
    status: OrderStatus
    order_value_inr: Optional[float] = Field(default=None, ge=0.0)
    escalation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    delayed_shipment: Optional[bool] = None
    error: Optional[str] = None



class ShipmentTrackingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^NYK-\d{5}$")
    status: str
    carrier: str
    tracking_number: str
    current_location: str
    estimated_delivery: str
    delayed_shipment: bool = False
    error: Optional[str] = None


class ReturnEligibilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^NYK-\d{5}$")
    eligible: bool
    category: Optional[str] = None
    days_since_delivery: Optional[int] = Field(default=None, ge=0)
    allowed_window_days: Optional[int] = Field(default=None, ge=0)
    reason: str
    error: Optional[str] = None


class ReturnRequestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(pattern=r"^NYK-\d{5}$")
    request_id: str
    status: str
    rma_code: str
    created_at: str
    reason: Optional[str] = None
    error: Optional[str] = None


class LoyaltyTier(str, Enum):
    SILVER = "Silver"
    GOLD = "Gold"
    PLATINUM = "Platinum"


class LoyaltyStatusResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    tier: LoyaltyTier
    points_balance: int = Field(ge=0)
    lifetime_spend_inr: float = Field(ge=0.0)
    error: Optional[str] = None


class EscalationPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EscalationCategory(str, Enum):
    POLICY_UNCERTAINTY = "policy_uncertainty"
    ORDER_DELAY = "order_delay"
    ORDER_NOT_FOUND = "order_not_found"
    CUSTOMER_REQUEST = "customer_request"
    GUARDRAIL_FLAG = "guardrail_flag"


class EscalationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_human: bool
    escalation_id: str = Field(min_length=1)
    category: EscalationCategory
    priority: EscalationPriority
    reason: str = Field(min_length=1)
    conversation_context: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    order_id: Optional[str] = None
    escalation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trace_id: str = Field(min_length=1)

    @field_validator("escalation_id", "reason", "conversation_context", "recommended_action", "trace_id")
    @classmethod
    def validate_non_blank_string(cls, val: str) -> str:
        if not val.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return val


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_type: ResponseType
    answer: str = Field(min_length=1)
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    trace_id: str = Field(min_length=1)
    escalation_payload: Optional[EscalationPayload] = None

    @field_validator("answer")
    @classmethod
    def validate_answer_not_blank(cls, val: str) -> str:
        if not val.strip():
            raise ValueError("Answer cannot be empty or whitespace only")
        return val

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, val: List[str]) -> List[str]:
        for item in val:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Sources must be non-empty strings")
        return val

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, val: str) -> str:
        if not val.strip():
            raise ValueError("Trace ID cannot be empty or whitespace only")
        return val


PROTECTED_DOCUMENTS = {
    "cancellation_policy.md",
    "cod_refund_timelines.md",
    "damaged_item_claims.md",
    "delivery_sla.md",
    "escalation_matrix.md",
    "international_shipping.md",
    "loyalty_points.md",
    "payment_failure_retry.md",
    "return_window.md",
    "reverse_pickup.md",
    "size_exchange.md",
    "warranty_terms.md",
}


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=1000)
    thread_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, val: str) -> str:
        stripped = val.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or whitespace only")
        return stripped

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        stripped = val.strip()
        if not stripped:
            raise ValueError("Thread ID cannot be empty or whitespace only")
        if not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", stripped):
            raise ValueError("Thread ID must contain only alphanumeric characters, dashes, or underscores")
        return stripped


class AddDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1, max_length=64)
    topic: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=50000)

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, val: str) -> str:
        stripped = val.strip()
        if ".." in stripped or "/" in stripped or "\\" in stripped:
            raise ValueError("Path traversal sequences are strictly forbidden in doc_id")
        if not re.match(r"^[a-zA-Z0-9_\-]{1,64}(\.md)?$", stripped):
            raise ValueError("doc_id must be an alphanumeric identifier optionally ending in .md")
        normalized = stripped if stripped.endswith(".md") else f"{stripped}.md"
        if normalized.lower() in PROTECTED_DOCUMENTS:
            raise ValueError(f"Overwriting protected core knowledge base document '{normalized}' is strictly prohibited")
        return normalized

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, val: str) -> str:
        stripped = val.strip()
        if not stripped:
            raise ValueError("Topic cannot be empty or whitespace only")
        return stripped

    @field_validator("text")
    @classmethod
    def validate_text(cls, val: str) -> str:
        stripped = val.strip()
        if not stripped:
            raise ValueError("Document text cannot be empty or whitespace only")
        return stripped


class AddDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="indexed")
    indexed_chunks: int = Field(ge=0)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok")


class ApiErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    message: str


def extract_safe_trace_id(raw_data: Any, fallback_id: Optional[str] = None) -> str:
    if isinstance(raw_data, dict):
        candidate = raw_data.get("trace_id")
        if isinstance(candidate, str) and SAFE_TRACE_PATTERN.match(candidate.strip()):
            return candidate.strip()
    if fallback_id and SAFE_TRACE_PATTERN.match(fallback_id.strip()):
        return fallback_id.strip()
    return str(uuid.uuid4())


def create_fail_closed_fallback(trace_id: str) -> AgentResponse:
    return AgentResponse(
        response_type=ResponseType.FALLBACK,
        answer=FALLBACK_RESPONSE,
        sources=[],
        confidence=0.0,
        escalation_score=None,
        trace_id=trace_id,
    )


def validate_agent_response(
    data: Any,
    default_trace_id: Optional[str] = None,
) -> AgentResponse:
    safe_trace = extract_safe_trace_id(data, default_trace_id)
    if not isinstance(data, dict):
        return create_fail_closed_fallback(safe_trace)
    try:
        return AgentResponse.model_validate(data)
    except Exception:
        return create_fail_closed_fallback(safe_trace)


def validate_order_result(data: Any) -> OrderStatusResult:
    if not isinstance(data, dict):
        raise ValueError("Order result must be a dictionary")
    return OrderStatusResult.model_validate(data)


def validate_shipment_result(data: Any) -> ShipmentTrackingResult:
    if not isinstance(data, dict):
        raise ValueError("Shipment result must be a dictionary")
    return ShipmentTrackingResult.model_validate(data)


def validate_return_eligibility_result(data: Any) -> ReturnEligibilityResult:
    if not isinstance(data, dict):
        raise ValueError("Return eligibility result must be a dictionary")
    return ReturnEligibilityResult.model_validate(data)


def validate_return_request_result(data: Any) -> ReturnRequestResult:
    if not isinstance(data, dict):
        raise ValueError("Return request result must be a dictionary")
    return ReturnRequestResult.model_validate(data)


def validate_loyalty_result(data: Any) -> LoyaltyStatusResult:
    if not isinstance(data, dict):
        raise ValueError("Loyalty result must be a dictionary")
    return LoyaltyStatusResult.model_validate(data)


def run_schema_tests() -> None:
    print("================================================================================")
    print("        NYKAA ASSIST TASK 10 — PYDANTIC SCHEMA VALIDATION TEST SUITE            ")
    print("================================================================================")

    invalid_cases = [
        ("Missing response_type", {"answer": "Hi", "sources": [], "confidence": 1.0, "trace_id": "tr-1"}),
        ("Missing answer", {"response_type": "policy_answer", "sources": [], "confidence": 1.0, "trace_id": "tr-1"}),
        ("Missing sources", {"response_type": "policy_answer", "answer": "Hi", "confidence": 1.0, "trace_id": "tr-1"}),
        ("Missing confidence", {"response_type": "policy_answer", "answer": "Hi", "sources": [], "trace_id": "tr-1"}),
        ("Missing trace_id", {"response_type": "policy_answer", "answer": "Hi", "sources": [], "confidence": 1.0}),
        ("Invalid response_type", {"response_type": "unknown_type", "answer": "Hi", "sources": [], "confidence": 1.0, "trace_id": "tr-1"}),
        ("Negative confidence", {"response_type": "policy_answer", "answer": "Hi", "sources": [], "confidence": -0.1, "trace_id": "tr-1"}),
        ("Confidence above 1.0", {"response_type": "policy_answer", "answer": "Hi", "sources": [], "confidence": 1.5, "trace_id": "tr-1"}),
        ("Negative escalation_score", {"response_type": "order_status", "answer": "Hi", "sources": [], "confidence": 1.0, "escalation_score": -0.2, "trace_id": "tr-1"}),
        ("Escalation score above 1.0", {"response_type": "order_status", "answer": "Hi", "sources": [], "confidence": 1.0, "escalation_score": 1.2, "trace_id": "tr-1"}),
        ("Empty answer string", {"response_type": "policy_answer", "answer": "", "sources": [], "confidence": 1.0, "trace_id": "tr-1"}),
        ("Whitespace only answer", {"response_type": "policy_answer", "answer": "   ", "sources": [], "confidence": 1.0, "trace_id": "tr-1"}),
        ("Forbidden unexpected field", {"response_type": "policy_answer", "answer": "Hi", "sources": [], "confidence": 1.0, "trace_id": "tr-1", "extra_hack": "malicious"}),
        ("Malformed sources item", {"response_type": "policy_answer", "answer": "Hi", "sources": [123], "confidence": 1.0, "trace_id": "tr-1"}),
    ]

    print("\n--- 1. Testing Deliberately Malformed AgentResponse Payloads ---")
    for label, payload in invalid_cases:
        failed = False
        try:
            AgentResponse.model_validate(payload)
        except ValidationError:
            failed = True
        print(f"[{'REJECTED' if failed else 'UNEXPECTED_PASS'}] {label}")
        assert failed, f"Payload should have been rejected by Pydantic: {label}"

    print("\n--- 2. Testing Deliberately Malformed OrderStatusResult Payloads ---")
    invalid_order_cases = [
        ("Invalid order ID pattern", {"record_id": "INVALID-123", "status": "Placed", "order_value_inr": 100.0, "escalation_score": 0.5}),
        ("Invalid order status enum", {"record_id": "NYK-00001", "status": "FabricatedStatus", "order_value_inr": 100.0, "escalation_score": 0.5}),
        ("Negative order value", {"record_id": "NYK-00001", "status": "Placed", "order_value_inr": -50.0, "escalation_score": 0.5}),
        ("Escalation score above 1.0", {"record_id": "NYK-00001", "status": "Placed", "order_value_inr": 100.0, "escalation_score": 1.8}),
        ("Forbidden unexpected field in order", {"record_id": "NYK-00001", "status": "Placed", "secret": "injected"}),
    ]
    for label, payload in invalid_order_cases:
        failed = False
        try:
            OrderStatusResult.model_validate(payload)
        except ValidationError:
            failed = True
        print(f"[{'REJECTED' if failed else 'UNEXPECTED_PASS'}] {label}")
        assert failed, f"Order payload should have been rejected by Pydantic: {label}"

    print("\n--- 3. Testing Fail-Closed Independent Fallback Generation ---")
    malformed_poisoned = {
        "response_type": "corrupted",
        "answer": "POISONED_SECRET_DATA_LEAK",
        "sources": ["malicious_source.md"],
        "confidence": 99.0,
        "escalation_score": -99.0,
        "trace_id": "trace-clean-123",
        "extra_leak": "db_password",
    }
    fail_closed_resp = validate_agent_response(malformed_poisoned)
    dumped = fail_closed_resp.model_dump()
    print(f"Fail-Closed Result Type: {dumped['response_type']}")
    print(f"Fail-Closed Answer     : {dumped['answer'][:60]}...")
    print(f"Fail-Closed Trace ID   : {dumped['trace_id']}")
    assert dumped["response_type"] == "fallback"
    assert dumped["answer"] == FALLBACK_RESPONSE
    assert dumped["sources"] == []
    assert dumped["confidence"] == 0.0
    assert dumped["escalation_score"] is None
    assert dumped["trace_id"] == "trace-clean-123"
    assert "POISONED_SECRET_DATA_LEAK" not in dumped["answer"]
    assert "extra_leak" not in dumped
    print("[PASSED] Fail-closed fallback constructed independently with zero data leakage!")

    non_dict_resp = validate_agent_response("STRING_NOT_A_DICT")
    assert non_dict_resp.response_type == ResponseType.FALLBACK
    assert non_dict_resp.answer == FALLBACK_RESPONSE
    print("[PASSED] Non-dict inputs safely fail closed to fallback!")

    print("\n--- 4. Testing All 5 Legitimate Customer Response Types ---")
    valid_cases = [
        (
            "Policy Answer",
            {
                "response_type": "policy_answer",
                "answer": "Customers can return unused beauty items within 15 days.",
                "sources": ["return_window.md"],
                "confidence": 0.544,
                "escalation_score": None,
                "trace_id": "tr-policy-1",
            },
        ),
        (
            "Order Status",
            {
                "response_type": "order_status",
                "answer": "Order NYK-00001 is currently placed with an order value of INR 2301.65.",
                "sources": [],
                "confidence": 1.0,
                "escalation_score": 0.693,
                "trace_id": "tr-order-1",
            },
        ),
        (
            "Missing Order (Not Found)",
            {
                "response_type": "order_status",
                "answer": "Order NYK-99999 was not found in our records. Please verify the order number.",
                "sources": [],
                "confidence": 1.0,
                "escalation_score": None,
                "trace_id": "tr-missing-1",
            },
        ),
        (
            "Grounded Fallback",
            {
                "response_type": "fallback",
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "confidence": 0.163,
                "escalation_score": None,
                "trace_id": "tr-fallback-1",
            },
        ),
        (
            "Guardrail Injection Block",
            {
                "response_type": "guardrail_block",
                "answer": "I cannot process this request as it violates our security policies. I am designed to assist exclusively with Nykaa customer support and order inquiries.",
                "sources": [],
                "confidence": 0.0,
                "escalation_score": None,
                "trace_id": "tr-guardrail-1",
            },
        ),
    ]
    for label, payload in valid_cases:
        model = AgentResponse.model_validate(payload)
        dump = model.model_dump()
        assert dump["response_type"] == payload["response_type"]
        assert dump["answer"] == payload["answer"]
        assert dump["trace_id"] == payload["trace_id"]
        print(f"[VALIDATED] {label} -> {model.response_type.value} (trace: {model.trace_id})")

    print("\n--- 5. Testing Valid and Invalid EscalationPayload ---")
    valid_esc_dict = {
        "requires_human": True,
        "escalation_id": "esc-test-1",
        "category": "policy_uncertainty",
        "priority": "medium",
        "reason": "Knowledge gate fallback",
        "conversation_context": "User asked out-of-scope query",
        "recommended_action": "Review customer policy question and provide manual guidance",
        "order_id": None,
        "escalation_score": None,
        "confidence": 0.2,
        "trace_id": "tr-esc-1",
    }
    esc_model = EscalationPayload.model_validate(valid_esc_dict)
    assert esc_model.requires_human is True
    assert esc_model.category == EscalationCategory.POLICY_UNCERTAINTY
    assert esc_model.priority == EscalationPriority.MEDIUM

    agent_with_esc = {
        "response_type": "fallback",
        "answer": FALLBACK_RESPONSE,
        "sources": [],
        "confidence": 0.2,
        "escalation_score": None,
        "trace_id": "tr-esc-1",
        "escalation_payload": valid_esc_dict,
    }
    agent_esc_model = AgentResponse.model_validate(agent_with_esc)
    assert agent_esc_model.escalation_payload is not None
    assert agent_esc_model.escalation_payload.escalation_id == "esc-test-1"

    invalid_esc_cases = [
        ("Missing requires_human", {**valid_esc_dict, "requires_human": None}),
        ("Invalid category", {**valid_esc_dict, "category": "invalid_cat"}),
        ("Invalid priority", {**valid_esc_dict, "priority": "urgent"}),
        ("Empty reason", {**valid_esc_dict, "reason": "   "}),
        ("Extra forbidden field", {**valid_esc_dict, "extra_hacker_field": 123}),
    ]
    for label, payload in invalid_esc_cases:
        failed = False
        try:
            EscalationPayload.model_validate(payload)
        except (ValidationError, ValueError):
            failed = True
        assert failed, f"Expected invalid escalation payload to fail: {label}"
        print(f"[REJECTED] {label}")

    print("\n================================================================================")
    print("         NYKAA ASSIST TASK 10 & 20 — ALL SCHEMA ASSERTIONS PASSED!               ")
    print("================================================================================")


if __name__ == "__main__":
    run_schema_tests()
