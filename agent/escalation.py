import re
import sys
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.guardrails import mask_pii
from agent.schema import (
    EscalationCategory,
    EscalationPayload,
    EscalationPriority,
    ResponseType,
)

ESCALATION_DELAY_THRESHOLD = 0.68
CONFIDENCE_FLOOR = 0.35
MAX_QUEUE_CAPACITY = 1000

ACTION_REVIEW_POLICY = "Review customer policy question and provide manual guidance"
ACTION_EXPEDITE_ORDER = "Expedite carrier delivery or initiate manual fulfillment trace"
ACTION_VERIFY_ORDER = "Verify order ID in warehouse database or request payment reference"
ACTION_ROUTE_LIVE_AGENT = "Route customer to live support chat agent"
ACTION_MANUAL_VALIDATION_ERROR = "Review customer query manually due to internal validation error"
ACTION_INVESTIGATE_TIMEOUT = "Investigate operational backend latency/timeout and contact customer"

EXPLICIT_HUMAN_PATTERNS = [
    re.compile(r"\b(?:speak|talk|chat|connect|transfer)\s+(?:(?:me\s+)?to\s+|with\s+)?(?:a\s+|an\s+)?(?:human|agent|representative|executive|person|operator|specialist|team)\b", re.IGNORECASE),
    re.compile(r"\b(?:human|agent|representative|executive|live\s+agent|real\s+person)\s+(?:support|assistance|help)\b", re.IGNORECASE),
    re.compile(r"\b(?:talk|speak)\s+to\s+someone\b", re.IGNORECASE),
    re.compile(r"\bcall\s+(?:a\s+|an\s+)?(?:human|agent|representative|executive)\b", re.IGNORECASE),
    re.compile(r"\bcustomer\s+care\s+(?:executive|representative|agent|human)\b", re.IGNORECASE),
    re.compile(r"\bhuman\s+in\s+the\s+loop\b", re.IGNORECASE),
    re.compile(r"\bescalat(?:e|ion)\s+to\s+(?:a\s+|an\s+)?(?:human|agent|representative|executive|manager|support)\b", re.IGNORECASE),
    re.compile(r"\bwant\s+(?:a\s+|an\s+)?human\b", re.IGNORECASE),
    re.compile(r"\bneed\s+(?:a\s+|an\s+)?human\b", re.IGNORECASE),
]


def is_explicit_human_request(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    clean_text = text.strip()
    if not clean_text:
        return False
    for pattern in EXPLICIT_HUMAN_PATTERNS:
        if pattern.search(clean_text):
            return True
    return False


def sanitize_context_for_escalation(
    query: str,
    last_topic: Optional[str] = None,
    last_order: Optional[str] = None,
    draft_answer: Optional[str] = None,
) -> str:
    clean_query = mask_pii(query or "")
    parts = [f"User Query: {clean_query}"]
    context_tags = []
    if last_topic:
        context_tags.append(f"previous_topic={mask_pii(last_topic)}")
    if last_order:
        context_tags.append(f"previous_order={mask_pii(last_order)}")
    if context_tags:
        parts.append(f"Conversation Context: {', '.join(context_tags)}")
    if draft_answer:
        parts.append(f"Agent Draft Answer: {mask_pii(draft_answer)}")
    joined = "\n".join(parts)
    return mask_pii(joined)


def create_fail_closed_escalation_payload(
    trace_id: str,
    query: str,
    reason: Optional[str] = None,
) -> EscalationPayload:
    safe_trace = trace_id.strip() if trace_id and trace_id.strip() else str(uuid.uuid4())
    safe_query = mask_pii(query.strip()) if query and query.strip() else "Unavailable query"
    return EscalationPayload(
        requires_human=True,
        escalation_id=f"esc-{safe_trace}",
        category=EscalationCategory.POLICY_UNCERTAINTY,
        priority=EscalationPriority.HIGH,
        reason=reason or "Internal escalation evaluation validation failure: fail-closed safety trigger",
        conversation_context=safe_query,
        recommended_action=ACTION_MANUAL_VALIDATION_ERROR,
        order_id=None,
        escalation_score=None,
        confidence=0.0,
        trace_id=safe_trace,
    )


def evaluate_escalation(state: Dict[str, Any]) -> Tuple[bool, Optional[EscalationPayload]]:
    if state.get("is_blocked"):
        return False, None

    query = state.get("query", "")
    orig_query = state.get("original_query", "") or query
    rewritten_query = state.get("rewritten_query", "") or ""
    trace_id = state.get("trace_id") or str(uuid.uuid4())
    response_data = state.get("response") or {}
    route = state.get("route") or "policy"
    gate_decision = state.get("gate_decision")
    order_id = state.get("order_id") or state.get("last_order_id")
    last_topic = state.get("last_policy_topic")
    last_order = state.get("last_order_id")

    resp_type = response_data.get("response_type")
    answer = response_data.get("answer", "")
    esc_score = response_data.get("escalation_score")
    confidence = response_data.get("confidence")

    try:
        has_explicit_intent = (
            is_explicit_human_request(query)
            or is_explicit_human_request(orig_query)
            or is_explicit_human_request(rewritten_query)
        )

        if has_explicit_intent:
            category = EscalationCategory.CUSTOMER_REQUEST
            priority = EscalationPriority.HIGH
            reason = "Customer explicitly requested human intervention"
            recommended_action = ACTION_ROUTE_LIVE_AGENT
            requires_human = True
        elif route == "order":
            answer_lower = answer.lower()
            is_timeout_or_exhausted = (
                response_data.get("status") in ("Timeout", "Retry Exhausted", "Operation Failed")
                or "timed out" in answer_lower
                or "exhausted" in answer_lower
                or "operation failed" in answer_lower
            )
            is_not_found = (
                response_data.get("status") == "Not Found"
                or "not found" in answer_lower
            )
            if is_timeout_or_exhausted:
                category = EscalationCategory.POLICY_UNCERTAINTY
                priority = EscalationPriority.HIGH
                target_ord = order_id or "unknown"
                reason = f"Operational tool timeout or retry exhaustion for {target_ord}"
                recommended_action = ACTION_INVESTIGATE_TIMEOUT
                requires_human = True
            elif is_not_found:
                category = EscalationCategory.ORDER_NOT_FOUND
                priority = EscalationPriority.MEDIUM
                target_ord = order_id or "unknown"
                reason = f"Order record {target_ord} not found in database"
                recommended_action = ACTION_VERIFY_ORDER
                requires_human = True
            elif esc_score is not None and esc_score >= ESCALATION_DELAY_THRESHOLD:
                category = EscalationCategory.ORDER_DELAY
                priority = EscalationPriority.HIGH
                reason = f"Severe order delay risk detected with escalation score {esc_score:.3f} >= {ESCALATION_DELAY_THRESHOLD}"
                recommended_action = ACTION_EXPEDITE_ORDER
                requires_human = True
            else:
                return False, None
        else:
            answer_lower = answer.lower()
            is_timeout = (
                response_data.get("status") in ("Global Timeout", "Timeout", "Retry Exhausted")
                or "timed out" in answer_lower
                or "deadline exceeded" in answer_lower
            )
            is_fallback = (
                is_timeout
                or gate_decision == "FALLBACK"
                or resp_type == ResponseType.FALLBACK.value
                or resp_type == "fallback"
                or (confidence is not None and confidence < CONFIDENCE_FLOOR)
            )
            if is_fallback:
                category = EscalationCategory.POLICY_UNCERTAINTY
                priority = EscalationPriority.HIGH if is_timeout else EscalationPriority.MEDIUM
                reason = (
                    "Execution deadline exceeded or operational timeout"
                    if is_timeout
                    else "Knowledge Gate returned FALLBACK or low confidence score below 0.35"
                )
                recommended_action = (
                    ACTION_INVESTIGATE_TIMEOUT
                    if is_timeout
                    else ACTION_REVIEW_POLICY
                )
                requires_human = True
            else:
                return False, None

        if not requires_human:
            return False, None

        context_str = sanitize_context_for_escalation(
            query=query or orig_query,
            last_topic=last_topic,
            last_order=last_order,
            draft_answer=answer,
        )

        payload = EscalationPayload(
            requires_human=True,
            escalation_id=f"esc-{trace_id}",
            category=category,
            priority=priority,
            reason=reason,
            conversation_context=context_str,
            recommended_action=recommended_action,
            order_id=order_id,
            escalation_score=esc_score,
            confidence=confidence,
            trace_id=trace_id,
        )
        return True, payload

    except (ValidationError, Exception) as exc:
        fail_closed_payload = create_fail_closed_escalation_payload(
            trace_id=trace_id,
            query=query or orig_query,
            reason=f"Internal escalation schema validation failure: {str(exc)}",
        )
        return True, fail_closed_payload


class HumanSupportQueue:
    def __init__(self, max_size: int = MAX_QUEUE_CAPACITY):
        self.max_size = max_size
        self._items: deque[EscalationPayload] = deque()
        self._enqueued_ids: Set[str] = set()
        self._lock = threading.Lock()

    def enqueue(self, payload: EscalationPayload) -> bool:
        if not isinstance(payload, EscalationPayload):
            return False
        with self._lock:
            if payload.escalation_id in self._enqueued_ids:
                return False
            if len(self._items) >= self.max_size:
                evicted = self._items.popleft()
                self._enqueued_ids.discard(evicted.escalation_id)
            self._items.append(payload)
            self._enqueued_ids.add(payload.escalation_id)
            return True

    def dequeue(self) -> Optional[EscalationPayload]:
        with self._lock:
            if not self._items:
                return None
            item = self._items.popleft()
            self._enqueued_ids.discard(item.escalation_id)
            return item

    def peek(self) -> Optional[EscalationPayload]:
        with self._lock:
            if not self._items:
                return None
            return self._items[0]

    def list_pending(
        self,
        category: Optional[Union[EscalationCategory, str]] = None,
        priority: Optional[Union[EscalationPriority, str]] = None,
    ) -> List[EscalationPayload]:
        with self._lock:
            cat_val = category.value if isinstance(category, EscalationCategory) else category
            prio_val = priority.value if isinstance(priority, EscalationPriority) else priority
            results = []
            for item in self._items:
                if cat_val is not None and item.category.value != cat_val:
                    continue
                if prio_val is not None and item.priority.value != prio_val:
                    continue
                results.append(item)
            return results

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._enqueued_ids.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def __len__(self) -> int:
        return self.size()


default_support_queue = HumanSupportQueue(max_size=MAX_QUEUE_CAPACITY)


def get_support_queue() -> HumanSupportQueue:
    return default_support_queue


def escalation_node(
    state: Dict[str, Any],
    queue: Optional[HumanSupportQueue] = None,
) -> Dict[str, Any]:
    target_queue = queue if queue is not None else get_support_queue()
    existing_raw_payload = state.get("escalation_payload")
    active_trace_id = state.get("trace_id")

    if existing_raw_payload is not None:
        try:
            if isinstance(existing_raw_payload, dict) and existing_raw_payload.get("trace_id") == active_trace_id:
                reconstituted = EscalationPayload.model_validate(existing_raw_payload)
                target_queue.enqueue(reconstituted)
                return {"escalation_payload": reconstituted.model_dump(mode="json")}
            elif isinstance(existing_raw_payload, EscalationPayload) and existing_raw_payload.trace_id == active_trace_id:
                target_queue.enqueue(existing_raw_payload)
                return {"escalation_payload": existing_raw_payload.model_dump(mode="json")}
        except Exception:
            pass

    requires_human, payload = evaluate_escalation(state)
    if requires_human and payload is not None:
        target_queue.enqueue(payload)
        return {"escalation_payload": payload.model_dump(mode="json")}

    return {"escalation_payload": None}
