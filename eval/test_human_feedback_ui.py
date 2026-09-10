import os
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["MOCK_LLM"] = "1"
os.environ["USE_REAL_LLM"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st

from agent.feedback import (
    CandidateType,
    FeedbackRecord,
    FeedbackSubmission,
    FeedbackType,
    default_feedback_store,
    lookup_trace_context,
    process_feedback_submission,
    record_trace_context,
)
from agent.graph import run_agent
from agent.guardrails import mask_pii
from agent.schema import ResponseType
from streamlit_app import (
    extract_safe_metadata,
    init_session_state,
    reset_chat_session,
    submit_feedback_for_message,
)


def setup_mock_session(thread_id: str = None) -> str:
    tid = thread_id or f"test_fb_session_{uuid.uuid4().hex[:12]}"
    st.session_state.thread_id = tid
    st.session_state.messages = []
    st.session_state.show_tech_details = False
    st.session_state.pending_query = None
    st.session_state.recorded_feedback_traces = set()
    return tid


def test_positive_feedback_submission():
    print("\n--- Test: Feedback Submission (Helpful) ---")
    thread_id = setup_mock_session()

    query = "What is the return window for beauty products?"
    state = run_agent(query, thread_id=thread_id, return_state=True)
    raw_resp = state.get("response", {})
    metadata = extract_safe_metadata(state, raw_resp)
    trace_id = metadata["trace_id"]

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "assistant",
        "content": raw_resp.get("answer", ""),
        "metadata": metadata,
        "feedback": None,
    })

    record = submit_feedback_for_message(1, "helpful")

    assert record is not None, "Expected FeedbackRecord from submit_feedback_for_message"
    assert record.rating == 5, f"Expected rating 5 for helpful, got {record.rating}"
    assert record.feedback_type == FeedbackType.HELPFUL, f"Expected HELPFUL, got {record.feedback_type}"
    assert record.trace_id == trace_id
    assert record.query_id == thread_id

    msg = st.session_state.messages[1]
    assert msg["feedback"] is not None
    assert msg["feedback"]["status"] == "recorded"
    assert msg["feedback"]["type"] == "helpful"
    assert msg["feedback"]["rating"] == 5
    assert msg["feedback"]["feedback_id"] == record.feedback_id

    assert trace_id in st.session_state.recorded_feedback_traces
    print(f"  [PASSED] Positive feedback recorded: feedback_id={record.feedback_id}, rating=5")


def test_negative_feedback_submission():
    print("\n--- Test: Feedback Submission (Not Helpful) ---")
    thread_id = setup_mock_session()

    query = "What is the status of NYK-00001?"
    state = run_agent(query, thread_id=thread_id, return_state=True)
    raw_resp = state.get("response", {})
    metadata = extract_safe_metadata(state, raw_resp)
    trace_id = metadata["trace_id"]

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({
        "role": "assistant",
        "content": raw_resp.get("answer", ""),
        "metadata": metadata,
        "feedback": None,
    })

    record = submit_feedback_for_message(1, "not_helpful")

    assert record is not None, "Expected FeedbackRecord from submit_feedback_for_message"
    assert record.rating == 1, f"Expected rating 1 for not helpful, got {record.rating}"
    assert record.feedback_type == FeedbackType.NOT_HELPFUL
    assert record.trace_id == trace_id

    assert record.improvement_candidate is not None, "Expected improvement candidate for negative feedback"
    assert record.improvement_candidate.human_rating == 1

    msg = st.session_state.messages[1]
    assert msg["feedback"]["status"] == "recorded"
    assert msg["feedback"]["type"] == "not_helpful"
    assert msg["feedback"]["rating"] == 1
    print(f"  [PASSED] Negative feedback recorded: feedback_id={record.feedback_id}")


def test_correct_response_association():
    print("\n--- Test: Correct Response-to-Feedback Association ---")
    thread_id = setup_mock_session()

    q1 = "Hi"
    s1 = run_agent(q1, thread_id=thread_id, return_state=True)
    r1 = s1.get("response", {})
    m1 = extract_safe_metadata(s1, r1)
    t1 = m1["trace_id"]
    st.session_state.messages.append({"role": "user", "content": q1})
    st.session_state.messages.append({"role": "assistant", "content": r1.get("answer"), "metadata": m1, "feedback": None})

    q2 = "What is the status of NYK-00005?"
    s2 = run_agent(q2, thread_id=thread_id, return_state=True)
    r2 = s2.get("response", {})
    m2 = extract_safe_metadata(s2, r2)
    t2 = m2["trace_id"]
    st.session_state.messages.append({"role": "user", "content": q2})
    st.session_state.messages.append({"role": "assistant", "content": r2.get("answer"), "metadata": m2, "feedback": None})

    q3 = "Can I return open cosmetics?"
    s3 = run_agent(q3, thread_id=thread_id, return_state=True)
    r3 = s3.get("response", {})
    m3 = extract_safe_metadata(s3, r3)
    t3 = m3["trace_id"]
    st.session_state.messages.append({"role": "user", "content": q3})
    st.session_state.messages.append({"role": "assistant", "content": r3.get("answer"), "metadata": m3, "feedback": None})

    rec1 = submit_feedback_for_message(1, "helpful")
    rec2 = submit_feedback_for_message(3, "not_helpful")

    assert rec1.trace_id == t1, f"Turn 1 trace mismatch: {rec1.trace_id} != {t1}"
    assert rec2.trace_id == t2, f"Turn 2 trace mismatch: {rec2.trace_id} != {t2}"
    assert rec1.answer_snapshot == r1.get("answer")
    assert "NYK-00005" in rec2.answer_snapshot

    assert st.session_state.messages[5]["feedback"] is None
    print("  [PASSED] Multi-turn messages strictly retain their own separate feedback associations.")


def test_duplicate_prevention():
    print("\n--- Test: Duplicate Feedback Prevention ---")
    thread_id = setup_mock_session()

    q = "Where is my order NYK-00001?"
    state = run_agent(q, thread_id=thread_id, return_state=True)
    raw_resp = state.get("response", {})
    metadata = extract_safe_metadata(state, raw_resp)
    trace_id = metadata["trace_id"]

    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.messages.append({
        "role": "assistant",
        "content": raw_resp.get("answer"),
        "metadata": metadata,
        "feedback": None,
    })

    rec_first = submit_feedback_for_message(1, "helpful")
    assert rec_first is not None
    assert rec_first.feedback_id

    rec_second = submit_feedback_for_message(1, "helpful")
    assert rec_second is None, "Duplicate submission must return None"

    rec_third = submit_feedback_for_message(1, "not_helpful")
    assert rec_third is None, "Opposite duplicate submission must return None"

    records = [r for r in default_feedback_store.list_feedback() if r.trace_id == trace_id]
    assert len(records) == 1, f"Expected exactly 1 record in store, found {len(records)}"
    print(f"  [PASSED] Accidental duplicate submissions prevented: exactly 1 record for trace {trace_id}")


def test_pii_scrubbing():
    print("\n--- Test: PII Scrubbing in Feedback Pipeline ---")
    thread_id = setup_mock_session()

    raw_pii_answer = (
        "Order delivered to Priya Mehta at Flat 302, Palm Grove. "
        "Email: priya.mehta@example.com, Phone: 9876543210, Card: 4111-2222-3333-4444."
    )
    fake_trace = f"trace-pii-{uuid.uuid4().hex[:8]}"
    st.session_state.messages.append({
        "role": "assistant",
        "content": raw_pii_answer,
        "metadata": {"trace_id": fake_trace, "route": "order"},
        "feedback": None,
    })

    rec = submit_feedback_for_message(0, "helpful")
    assert rec is not None

    assert "priya.mehta@example.com" not in rec.answer_snapshot
    assert "9876543210" not in rec.answer_snapshot
    assert "4111-2222-3333-4444" not in rec.answer_snapshot

    sensitive_items = ["priya.mehta@example.com", "9876543210", "4111-2222-3333-4444"]
    assert default_feedback_store.verify_no_raw_pii(sensitive_items) is True
    print("  [PASSED] PII strictly scrubbed and verified absent in feedback persistence.")


def test_feedback_persistence():
    print("\n--- Test: Feedback Persistence in SQLite ---")
    thread_id = setup_mock_session()

    fake_trace = f"trace-persist-{uuid.uuid4().hex[:8]}"
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Beauty products carry a 15-day return window.",
        "metadata": {"trace_id": fake_trace, "route": "policy"},
        "feedback": None,
    })

    rec = submit_feedback_for_message(0, "helpful")
    fb_id = rec.feedback_id

    fetched = default_feedback_store.get_feedback(fb_id)
    assert fetched is not None, f"Could not find feedback {fb_id} in database"
    assert fetched.feedback_id == fb_id
    assert fetched.trace_id == fake_trace
    assert fetched.rating == 5
    assert fetched.feedback_type == FeedbackType.HELPFUL
    assert fetched.pii_scrubbed is True
    print(f"  [PASSED] Feedback successfully retrieved from SQLite: {fetched.feedback_id}")


def test_streamlit_rerun_behavior():
    print("\n--- Test: Streamlit Rerun Safety ---")
    thread_id = setup_mock_session()

    st.session_state.messages.append({
        "role": "assistant",
        "content": "Order NYK-00001 is placed.",
        "metadata": {"trace_id": f"trace-rerun-{uuid.uuid4().hex[:8]}", "route": "order"},
        "feedback": None,
    })

    msg_count_before = len(st.session_state.messages)

    rec = submit_feedback_for_message(0, "helpful")
    assert rec is not None

    msg_count_after = len(st.session_state.messages)
    assert msg_count_before == msg_count_after, "Rerun must not duplicate messages"
    assert st.session_state.messages[0]["feedback"]["status"] == "recorded"
    assert st.session_state.pending_query is None, "Rerun must not create pending queries"
    print("  [PASSED] Streamlit rerun preserves state without message duplication or unintended execution.")


def main():
    print("=" * 80)
    print("   NYKAA ASSIST — STREAMLIT HUMAN FEEDBACK UI TEST SUITE")
    print("=" * 80)

    test_positive_feedback_submission()
    test_negative_feedback_submission()
    test_correct_response_association()
    test_duplicate_prevention()
    test_pii_scrubbing()
    test_feedback_persistence()
    test_streamlit_rerun_behavior()

    print("\n" + "=" * 80)
    print("   ALL STREAMLIT HUMAN FEEDBACK UI TESTS PASSED CLEANLY (100%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
