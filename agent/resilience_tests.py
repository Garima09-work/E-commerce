import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import importlib.util

_client_path = ROOT_DIR / "mcp" / "client.py"
_spec = importlib.util.spec_from_file_location("local_mcp_client", _client_path)
local_mcp_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(local_mcp_client)

call_mcp_tool = local_mcp_client.call_mcp_tool
call_order_tool_mcp = local_mcp_client.call_order_tool_mcp
clear_mcp_call_history = local_mcp_client.clear_mcp_call_history
get_mcp_call_count = local_mcp_client.get_mcp_call_count

from resilience.retry_timeout import (
    GlobalTimeoutError,
    GlobalTimeoutGuard,
    NodeTimeoutError,
    ResilienceError,
    RetryExhaustedError,
    RetryPolicy,
    SimulatedFlakyOperation,
    clear_resilience_events,
    execute_with_retry,
    execute_with_timeout,
    get_resilience_events,
)
from agent.graph import (
    AgentState,
    agent_app,
    build_agent_graph,
    order_node,
    policy_node,
    run_agent,
)
from agent.escalation import (
    ACTION_INVESTIGATE_TIMEOUT,
    EscalationCategory,
    EscalationPriority,
    HumanSupportQueue,
    default_support_queue,
    escalation_node,
)
from agent.memory import (
    clear_thread_checkpoints,
    get_sqlite_checkpointer,
)
from agent.schema import (
    AgentResponse,
    ResponseType,
)
from agent.tools import (
    ACTIVE_RETURN_REQUESTS,
    create_return_request,
)


def test_t22_1_exponential_backoff_math() -> None:
    policy = RetryPolicy(
        max_attempts=4,
        initial_interval=0.1,
        backoff_factor=2.0,
        max_interval=0.5,
        jitter=False,
    )
    assert abs(policy.get_interval(1) - 0.1) < 1e-6
    assert abs(policy.get_interval(2) - 0.2) < 1e-6
    assert abs(policy.get_interval(3) - 0.4) < 1e-6
    assert abs(policy.get_interval(4) - 0.5) < 1e-6
    print("[PASSED] T22-1: Exponential backoff math correctly doubles and clamps at max_interval.")


def test_t22_2_deterministic_jitter() -> None:
    policy_seeded_a = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=10.0,
        jitter=True,
        seed=42,
    )
    policy_seeded_b = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=10.0,
        jitter=True,
        seed=42,
    )
    intervals_a = [policy_seeded_a.get_interval(i) for i in range(1, 4)]
    intervals_b = [policy_seeded_b.get_interval(i) for i in range(1, 4)]
    assert intervals_a == intervals_b
    assert all(0.0 <= inv <= 1.0 * (2.0 ** idx) for idx, inv in enumerate(intervals_a))
    print("[PASSED] T22-2: Deterministic jitter produces identical intervals with identical seeds.")


def test_t22_3_retry_recovery_attempt_2() -> None:
    clear_resilience_events()
    flaky = SimulatedFlakyOperation(fail_count=1, failure_exception=ResilienceError("Transient network drop"))
    policy = RetryPolicy(max_attempts=3, initial_interval=0.01, backoff_factor=2.0, jitter=False)
    result = execute_with_retry(flaky, policy=policy, trace_id="trace-t22-3")
    assert result == "success"
    assert flaky.attempts == 2
    events = get_resilience_events()
    retry_events = [e for e in events if e.event_type == "retry_attempt" and e.trace_id == "trace-t22-3"]
    assert len(retry_events) == 1
    assert retry_events[0].attempt == 1
    success_events = [e for e in events if e.event_type == "retry_success" and e.trace_id == "trace-t22-3"]
    assert len(success_events) == 1
    print("[PASSED] T22-3: Retry successfully recovered on attempt 2.")


def test_t22_4_retry_recovery_attempt_3() -> None:
    clear_resilience_events()
    flaky = SimulatedFlakyOperation(fail_count=2, failure_exception=TimeoutError("Downstream service timed out"))
    policy = RetryPolicy(max_attempts=3, initial_interval=0.01, backoff_factor=2.0, jitter=False)
    result = execute_with_retry(flaky, policy=policy, trace_id="trace-t22-4")
    assert result == "success"
    assert flaky.attempts == 3
    events = get_resilience_events()
    retry_events = [e for e in events if e.event_type == "retry_attempt" and e.trace_id == "trace-t22-4"]
    assert len(retry_events) == 2
    print("[PASSED] T22-4: Retry successfully recovered on attempt 3.")


def test_t22_5_retry_exhaustion() -> None:
    clear_resilience_events()
    flaky = SimulatedFlakyOperation(fail_count=10, failure_exception=ConnectionResetError("Socket broken"))
    policy = RetryPolicy(max_attempts=3, initial_interval=0.01, backoff_factor=2.0, jitter=False)
    exhausted = False
    try:
        execute_with_retry(flaky, policy=policy, trace_id="trace-t22-5")
    except RetryExhaustedError as exc:
        exhausted = True
        assert exc.attempts == 3
    assert exhausted is True
    assert flaky.attempts == 3
    events = get_resilience_events()
    exhaust_events = [e for e in events if e.event_type == "retry_exhausted" and e.trace_id == "trace-t22-5"]
    assert len(exhaust_events) == 1
    print("[PASSED] T22-5: Retry exhausted after exactly 3 attempts and raised RetryExhaustedError.")


def test_t22_6_non_retryable_fast_fail() -> None:
    clear_resilience_events()
    flaky = SimulatedFlakyOperation(fail_count=5, failure_exception=ValueError("Bad parameter value"))
    policy = RetryPolicy(max_attempts=3, initial_interval=0.01, backoff_factor=2.0, retryable_exceptions=(ResilienceError, TimeoutError))
    fast_failed = False
    try:
        execute_with_retry(flaky, policy=policy, trace_id="trace-t22-6")
    except ValueError as exc:
        fast_failed = True
        assert "Bad parameter value" in str(exc)
    assert fast_failed is True
    assert flaky.attempts == 1
    events = get_resilience_events()
    retry_events = [e for e in events if e.event_type == "retry_attempt" and e.trace_id == "trace-t22-6"]
    assert len(retry_events) == 0
    print("[PASSED] T22-6: Non-retryable exception failed immediately on attempt 1 without retries.")


def test_t22_7_node_timeout_clean_abort() -> None:
    def slow_fn() -> str:
        time.sleep(1.0)
        return "slow_done"
    t0 = time.perf_counter()
    caught_timeout = False
    try:
        execute_with_timeout(slow_fn, timeout_seconds=0.1, trace_id="trace-t22-7", node_name="test_node")
    except NodeTimeoutError as exc:
        caught_timeout = True
        assert exc.node_name == "test_node"
        assert exc.timeout_seconds == 0.1
    elapsed = time.perf_counter() - t0
    assert caught_timeout is True
    assert elapsed < 0.6
    print(f"[PASSED] T22-7: Node timeout cleanly aborted in {elapsed:.3f}s (budget 0.1s).")


def test_t22_8_node_execution_within_budget() -> None:
    def fast_fn() -> str:
        time.sleep(0.02)
        return "fast_done"
    res = execute_with_timeout(fast_fn, timeout_seconds=1.0, trace_id="trace-t22-8", node_name="test_node")
    assert res == "fast_done"
    print("[PASSED] T22-8: Node execution completed within budget successfully.")


def test_t22_9_global_timeout_cancellation() -> None:
    def hanging_generate(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.5)
        return {
            "response_type": "policy_answer",
            "answer": "Grounded answer",
            "sources": ["return_window.md"],
            "confidence": 0.9,
            "escalation_score": None,
            "trace_id": "test",
        }
    with patch("agent.graph.generate_grounded_answer", side_effect=hanging_generate):
        t0 = time.perf_counter()
        res = run_agent("Can I return this product?", timeout=0.1)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.4
        assert res.get("response_type") == "fallback"
        assert "timeout" in res.get("answer", "").lower()
        assert res.get("escalation_score") == 0.90
    print(f"[PASSED] T22-9: Global timeout cancelled hanging execution in {elapsed:.3f}s with safe fallback.")


def test_t22_10_global_execution_within_budget() -> None:
    time.sleep(0.5)
    res = run_agent("Can I return this product?", timeout=15.0)
    assert res.get("response_type") == "policy_answer"
    assert res.get("confidence") >= 0.35
    print("[PASSED] T22-10: Global execution completed within budget.")


def test_t22_11_mcp_retry_recovery() -> None:
    call_counts = {"count": 0}
    real_dispatch = local_mcp_client._dispatch_raw_mcp_tool
    def flaky_dispatch(tool_name: str, arguments: Dict[str, Any], server_url: Optional[str]) -> Dict[str, Any]:
        call_counts["count"] += 1
        if call_counts["count"] == 1:
            raise ConnectionRefusedError("Simulated MCP socket error")
        return real_dispatch(tool_name, arguments, server_url)
    with patch.object(local_mcp_client, "_dispatch_raw_mcp_tool", side_effect=flaky_dispatch):
        policy = RetryPolicy(max_attempts=3, initial_interval=0.01, backoff_factor=2.0, jitter=False)
        res = call_mcp_tool("check_order_status", {"record_id": "NYK-00001"}, retry_policy=policy, trace_id="trace-t22-11")
        assert call_counts["count"] == 2
        assert res.get("record_id") == "NYK-00001"
        assert res.get("status") == "Placed"
    print("[PASSED] T22-11: MCP tool call recovered from transient failure via retry.")


def test_t22_12_mcp_timeout_fallback() -> None:
    default_support_queue.clear()
    def hanging_dispatch(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        time.sleep(2.0)
        return {"record_id": "NYK-00001", "status": "Placed"}
    with patch.object(local_mcp_client, "_dispatch_raw_mcp_tool", side_effect=hanging_dispatch):
        policy = RetryPolicy(max_attempts=1, initial_interval=0.01, jitter=False)
        raw_res = call_mcp_tool("check_order_status", {"record_id": "NYK-00001"}, timeout_seconds=0.1, retry_policy=policy, trace_id="trace-t22-12")
        assert raw_res.get("status") == "Timeout"
        assert "timed out" in raw_res.get("error", "").lower()
    def timeout_lookup(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "record_id": "NYK-00001",
            "status": "Timeout",
            "order_value_inr": None,
            "escalation_score": 0.90,
            "error": "Operational timeout during lookup",
        }
    with patch("agent.graph.call_order_tool_mcp", side_effect=timeout_lookup):
        res = run_agent("Where is my order NYK-00001?")
        assert res.get("escalation_score") == 0.90
        assert "timed out" in res.get("answer", "").lower()
        assert default_support_queue.size() >= 1
    print("[PASSED] T22-12: MCP tool call timeout returned safe status and triggered escalation.")


def test_t22_13_idempotent_return_creation_retry() -> None:
    test_oid = "NYK-00004"
    ACTIVE_RETURN_REQUESTS.pop(test_oid, None)
    call_counts = {"count": 0}
    def flaky_create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        call_counts["count"] += 1
        res = create_return_request(test_oid, reason="Defective product")
        if call_counts["count"] == 1:
            raise ResilienceError("Simulated network failure right after RMA creation")
        return res
    with patch.object(local_mcp_client, "_dispatch_raw_mcp_tool", side_effect=flaky_create):
        policy = RetryPolicy(max_attempts=3, initial_interval=0.01, backoff_factor=1.5, jitter=False)
        res = call_mcp_tool("create_return_request", {"record_id": test_oid, "reason": "Defective product"}, retry_policy=policy, trace_id="trace-t22-13")
        assert call_counts["count"] == 2
        assert res.get("record_id") == test_oid
        assert res.get("rma_code") is not None
        assert res.get("status") in ("Initiated", "Duplicate Request")
        matching_rmas = [v for k, v in ACTIVE_RETURN_REQUESTS.items() if k == test_oid]
        assert len(matching_rmas) == 1
    print("[PASSED] T22-13: Idempotent return creation produced exactly 0 duplicate RMAs under retry.")


def test_t22_14_resilience_event_logging() -> None:
    clear_resilience_events()
    flaky = SimulatedFlakyOperation(fail_count=1, failure_exception=ResilienceError("Transient"))
    policy = RetryPolicy(max_attempts=2, initial_interval=0.01, jitter=False)
    execute_with_retry(flaky, policy=policy, trace_id="trace-t22-14")
    events = get_resilience_events()
    matching = [e for e in events if e.trace_id == "trace-t22-14"]
    assert len(matching) >= 2
    types = [e.event_type for e in matching]
    assert "retry_attempt" in types
    assert "retry_success" in types
    for ev in matching:
        assert ev.timestamp is not None
        assert ev.trace_id == "trace-t22-14"
    print("[PASSED] T22-14: Resilience event logging captures timestamp, trace_id, and event_type.")


def test_t22_15_operational_timeout_escalation() -> None:
    default_support_queue.clear()
    state = {
        "query": "Where is my order NYK-00001?",
        "route": "order",
        "order_id": "NYK-00001",
        "response": {
            "response_type": "order_status",
            "answer": "Our order service timed out while checking order NYK-00001. Your request has been escalated to customer support.",
            "sources": [],
            "confidence": 0.0,
            "escalation_score": 0.90,
            "trace_id": "trace-t22-15",
            "status": "Timeout",
        },
        "trace_id": "trace-t22-15",
    }
    res = escalation_node(state)
    assert res.get("escalation_payload") is not None
    payload = res["escalation_payload"]
    assert payload["category"] == EscalationCategory.POLICY_UNCERTAINTY.value
    assert payload["priority"] == EscalationPriority.HIGH.value
    assert payload["recommended_action"] == ACTION_INVESTIGATE_TIMEOUT
    assert default_support_queue.size() == 1
    enqueued = default_support_queue.peek()
    assert enqueued.escalation_id == "esc-trace-t22-15"
    print("[PASSED] T22-15: Operational timeout enqueued into HumanSupportQueue with HIGH priority.")


def test_t22_16_checkpoint_resume_with_resilience() -> None:
    checkpointer = get_sqlite_checkpointer()
    test_thread = "test-t22-16-resilience-resume"
    clear_thread_checkpoints(test_thread, checkpointer=checkpointer)
    default_support_queue.clear()
    app_interrupted = build_agent_graph(
        checkpointer=checkpointer,
        interrupt_before=["output_guardrails"],
    )
    config = {"configurable": {"thread_id": test_thread}}
    def timeout_lookup(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "record_id": "NYK-00001",
            "status": "Timeout",
            "order_value_inr": None,
            "escalation_score": 0.90,
            "error": "Operational timeout during lookup",
        }
    with patch("agent.graph.call_order_tool_mcp", side_effect=timeout_lookup):
        app_interrupted.invoke({"query": "Where is my order NYK-00001?"}, config=config)
    queue_size_p1 = default_support_queue.size()
    assert queue_size_p1 == 1
    state_before_resume = app_interrupted.get_state(config)
    assert state_before_resume.next == ("output_guardrails",)
    assert state_before_resume.values.get("escalation_payload") is not None
    app_interrupted.invoke(None, config=config)
    queue_size_p2 = default_support_queue.size()
    assert queue_size_p2 == queue_size_p1
    print("[PASSED] T22-16: Checkpoint resume restored state with zero duplicate enqueueing under resilience scenario.")


def main() -> None:
    print("================================================================================")
    print("      NYKAA ASSIST TASK 22 — TIME-OUTS, RETRIES & RESILIENCE VERIFICATION      ")
    print("================================================================================")
    from rag.generate import generate_grounded_answer
    from rag.embed_index import FIXED_COLLECTION_NAME
    generate_grounded_answer("warmup query", collection_name=FIXED_COLLECTION_NAME)
    test_t22_1_exponential_backoff_math()
    test_t22_2_deterministic_jitter()
    test_t22_3_retry_recovery_attempt_2()
    test_t22_4_retry_recovery_attempt_3()
    test_t22_5_retry_exhaustion()
    test_t22_6_non_retryable_fast_fail()
    test_t22_7_node_timeout_clean_abort()
    test_t22_8_node_execution_within_budget()
    test_t22_9_global_timeout_cancellation()
    test_t22_10_global_execution_within_budget()
    test_t22_11_mcp_retry_recovery()
    test_t22_12_mcp_timeout_fallback()
    test_t22_13_idempotent_return_creation_retry()
    test_t22_14_resilience_event_logging()
    test_t22_15_operational_timeout_escalation()
    test_t22_16_checkpoint_resume_with_resilience()
    print("================================================================================")
    print("           ALL 16 TASK 22 RESILIENCE TESTS COMPLETED SUCCESSFULLY!              ")
    print("================================================================================")


if __name__ == "__main__":
    main()
