import json
import os
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.guardrails import mask_pii
from agent.schema import SAFE_TRACE_PATTERN
from rag.generate import FALLBACK_RESPONSE
from service.logging_utils import DEFAULT_LOG_PATH, read_jsonl_logs

DEFAULT_FEEDBACK_DB_PATH = ROOT_DIR / "feedback.sqlite"


class FeedbackType(str, Enum):
    HELPFUL = "helpful"
    PARTIALLY_HELPFUL = "partially_helpful"
    NOT_HELPFUL = "not_helpful"
    INCORRECT = "incorrect"
    UNSAFE = "unsafe"
    IRRELEVANT = "irrelevant"
    OTHER = "other"


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ACTIONABLE = "ACTIONABLE"


class CandidateType(str, Enum):
    VERIFICATION_MISMATCH = "verification_mismatch"
    LOW_RATING = "low_rating"
    INCORRECT_OUTPUT = "incorrect_output"
    SAFETY_FLAG = "safety_flag"


class ImprovementCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(default_factory=lambda: f"cand-{uuid.uuid4().hex[:12]}")
    candidate_type: CandidateType
    reason: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    verification_decision: Optional[str] = None
    human_rating: int = Field(ge=1, le=5)
    recommended_action: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FeedbackSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(ge=1, le=5)
    feedback_type: FeedbackType
    comment: Optional[str] = Field(default=None, max_length=2000)
    query_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, val: str) -> str:
        stripped = val.strip()
        if not stripped or not SAFE_TRACE_PATTERN.match(stripped):
            raise ValueError("Invalid trace_id pattern")
        return stripped

    @field_validator("comment")
    @classmethod
    def validate_comment(cls, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        stripped = val.strip()
        if not stripped:
            return None
        return stripped

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        stripped = val.strip()
        if not stripped:
            return None
        if not SAFE_TRACE_PATTERN.match(stripped):
            raise ValueError("Invalid query_id pattern")
        return stripped


class FeedbackRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: str = Field(default_factory=lambda: f"fbk-{uuid.uuid4().hex[:12]}")
    trace_id: str = Field(min_length=1)
    query_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rating: int = Field(ge=1, le=5)
    feedback_type: FeedbackType
    comment: Optional[str] = None
    sanitized_comment: Optional[str] = None
    answer_snapshot: Optional[str] = None
    verification_status: Optional[str] = None
    verification_decision: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def sync_comment_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            c = data.get("comment")
            sc = data.get("sanitized_comment")
            if c is not None and sc is None:
                data["sanitized_comment"] = c
            elif sc is not None and c is None:
                data["comment"] = sc
        return data
    route: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    tool_used: Optional[str] = None
    reviewed: bool = False
    review_status: ReviewStatus = ReviewStatus.PENDING
    improvement_candidate: Optional[ImprovementCandidate] = None
    pii_scrubbed: bool = True
    reviewed_at: Optional[str] = None
    reviewer_notes: Optional[str] = None


class FeedbackReviewUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_status: ReviewStatus
    reviewer_notes: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("reviewer_notes")
    @classmethod
    def validate_notes(cls, val: Optional[str]) -> Optional[str]:
        if val is None:
            return None
        stripped = val.strip()
        return stripped if stripped else None


class FeedbackAcknowledgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "received"
    feedback_id: str
    trace_id: str
    message: str = "Thank you for your feedback."


_TRACE_CONTEXT_CACHE: Dict[str, Dict[str, Any]] = {}


def record_trace_context(
    trace_id: str,
    answer: Optional[str] = None,
    route: Optional[str] = None,
    tool_used: Optional[str] = None,
    evidence_ids: Optional[List[str]] = None,
    verification_decision: Optional[str] = None,
    verification_status: Optional[str] = None,
    query_id: Optional[str] = None,
) -> None:
    if not trace_id or not isinstance(trace_id, str):
        return
    safe_trace = trace_id.strip()
    if not SAFE_TRACE_PATTERN.match(safe_trace):
        return

    safe_answer = None
    if answer and isinstance(answer, str):
        safe_answer = mask_pii(answer)
        if (verification_decision or "").upper() == "REJECT" or (verification_status or "").upper() == "REJECT":
            safe_answer = FALLBACK_RESPONSE

    safe_evidence = [str(eid) for eid in (evidence_ids or []) if eid]

    entry = {
        "trace_id": safe_trace,
        "query_id": query_id,
        "answer_snapshot": safe_answer,
        "route": route,
        "tool_used": tool_used,
        "evidence_ids": safe_evidence,
        "verification_decision": (verification_decision or "").upper() or None,
        "verification_status": (verification_status or "").upper() or None,
    }

    if len(_TRACE_CONTEXT_CACHE) > 1000:
        first_key = next(iter(_TRACE_CONTEXT_CACHE))
        _TRACE_CONTEXT_CACHE.pop(first_key, None)

    _TRACE_CONTEXT_CACHE[safe_trace] = entry


def clear_trace_context_cache() -> None:
    _TRACE_CONTEXT_CACHE.clear()


def lookup_trace_context(
    trace_id: str,
    log_path: Union[str, Path] = DEFAULT_LOG_PATH,
) -> Dict[str, Any]:
    if not trace_id or not isinstance(trace_id, str):
        return {
            "trace_id": str(uuid.uuid4()),
            "query_id": None,
            "answer_snapshot": None,
            "route": "UNKNOWN",
            "tool_used": None,
            "evidence_ids": [],
            "verification_decision": "UNKNOWN",
            "verification_status": "UNKNOWN",
        }

    safe_trace = trace_id.strip()
    if safe_trace in _TRACE_CONTEXT_CACHE:
        return dict(_TRACE_CONTEXT_CACHE[safe_trace])

    recovered: Dict[str, Any] = {
        "trace_id": safe_trace,
        "query_id": None,
        "answer_snapshot": None,
        "route": "UNKNOWN",
        "tool_used": None,
        "evidence_ids": [],
        "verification_decision": "UNKNOWN",
        "verification_status": "UNKNOWN",
    }

    try:
        records = read_jsonl_logs(log_path)
        for r in records:
            if r.get("trace_id") != safe_trace:
                continue
            if r.get("event") == "answer_verification":
                recovered["verification_decision"] = r.get("decision", "UNKNOWN")
                recovered["verification_status"] = r.get("decision", "UNKNOWN")
                if "evidence_ids" in r and isinstance(r["evidence_ids"], list):
                    recovered["evidence_ids"] = r["evidence_ids"]
            if r.get("endpoint") == "/ask":
                if r.get("route"):
                    recovered["route"] = r.get("route")
                if r.get("tool_called"):
                    recovered["tool_used"] = r.get("tool_called")
                if r.get("sources") and not recovered["evidence_ids"]:
                    recovered["evidence_ids"] = r.get("sources", [])
    except Exception:
        pass

    return recovered


def generate_improvement_candidate(
    trace_id: str,
    rating: int,
    feedback_type: FeedbackType,
    verification_decision: Optional[str] = None,
) -> Optional[ImprovementCandidate]:
    dec = (verification_decision or "").upper()
    if dec == "PASS" and (rating <= 2 or feedback_type in (FeedbackType.INCORRECT, FeedbackType.NOT_HELPFUL)):
        return ImprovementCandidate(
            candidate_type=CandidateType.VERIFICATION_MISMATCH,
            reason=f"Verification decision was PASS but human feedback was rating={rating} ({feedback_type.value})",
            trace_id=trace_id,
            verification_decision=dec,
            human_rating=rating,
            recommended_action="review_verification_rule_and_evidence",
        )

    if rating <= 2:
        return ImprovementCandidate(
            candidate_type=CandidateType.LOW_RATING,
            reason=f"Customer satisfaction rating of {rating}/5 with feedback type '{feedback_type.value}'",
            trace_id=trace_id,
            verification_decision=dec if dec else None,
            human_rating=rating,
            recommended_action="evaluate_response_quality_and_retrieval",
        )

    if feedback_type == FeedbackType.INCORRECT:
        return ImprovementCandidate(
            candidate_type=CandidateType.INCORRECT_OUTPUT,
            reason="Customer indicated the response contained incorrect information",
            trace_id=trace_id,
            verification_decision=dec if dec else None,
            human_rating=rating,
            recommended_action="audit_retrieved_evidence_and_grounded_generator",
        )

    if feedback_type == FeedbackType.UNSAFE:
        return ImprovementCandidate(
            candidate_type=CandidateType.SAFETY_FLAG,
            reason="Customer flagged response as potentially unsafe or violating policy",
            trace_id=trace_id,
            verification_decision=dec if dec else None,
            human_rating=rating,
            recommended_action="audit_guardrails_and_safety_filters",
        )

    return None


class FeedbackStore:
    def __init__(self, db_path: Union[str, Path] = DEFAULT_FEEDBACK_DB_PATH) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA busy_timeout=30000;")
            except Exception:
                pass
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS human_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    query_id TEXT,
                    timestamp TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    feedback_type TEXT NOT NULL,
                    sanitized_comment TEXT,
                    answer_snapshot TEXT,
                    verification_status TEXT,
                    verification_decision TEXT,
                    route TEXT,
                    evidence_ids TEXT,
                    tool_used TEXT,
                    reviewed INTEGER NOT NULL DEFAULT 0,
                    review_status TEXT NOT NULL DEFAULT 'PENDING',
                    improvement_candidate TEXT,
                    pii_scrubbed INTEGER NOT NULL DEFAULT 1,
                    reviewed_at TEXT,
                    reviewer_notes TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_trace_id ON human_feedback(trace_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_status ON human_feedback(review_status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fb_rating ON human_feedback(rating)")
            conn.commit()

    def save_feedback(self, record: FeedbackRecord) -> FeedbackRecord:
        clean_comment = mask_pii(record.comment) if record.comment else None
        clean_snapshot = mask_pii(record.answer_snapshot) if record.answer_snapshot else None
        if (record.verification_decision or "").upper() == "REJECT" or (record.verification_status or "").upper() == "REJECT":
            clean_snapshot = FALLBACK_RESPONSE

        cand_json = None
        if record.improvement_candidate:
            cand_json = json.dumps(record.improvement_candidate.model_dump(mode="json"))

        ev_json = json.dumps(record.evidence_ids)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO human_feedback (
                    feedback_id, trace_id, query_id, timestamp, rating,
                    feedback_type, sanitized_comment, answer_snapshot,
                    verification_status, verification_decision, route,
                    evidence_ids, tool_used, reviewed, review_status,
                    improvement_candidate, pii_scrubbed, reviewed_at, reviewer_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.feedback_id,
                    record.trace_id,
                    record.query_id,
                    record.timestamp,
                    int(record.rating),
                    record.feedback_type.value,
                    clean_comment,
                    clean_snapshot,
                    record.verification_status,
                    record.verification_decision,
                    record.route,
                    ev_json,
                    record.tool_used,
                    1 if record.reviewed else 0,
                    record.review_status.value,
                    cand_json,
                    1 if record.pii_scrubbed else 0,
                    record.reviewed_at,
                    record.reviewer_notes,
                ),
            )
            conn.commit()

        return self.get_feedback(record.feedback_id) or record

    def _row_to_record(self, row: sqlite3.Row) -> FeedbackRecord:
        ev_ids = []
        if row["evidence_ids"]:
            try:
                ev_ids = json.loads(row["evidence_ids"])
            except Exception:
                ev_ids = []

        cand = None
        if row["improvement_candidate"]:
            try:
                cand_data = json.loads(row["improvement_candidate"])
                cand = ImprovementCandidate.model_validate(cand_data)
            except Exception:
                cand = None

        return FeedbackRecord(
            feedback_id=row["feedback_id"],
            trace_id=row["trace_id"],
            query_id=row["query_id"],
            timestamp=row["timestamp"],
            rating=int(row["rating"]),
            feedback_type=FeedbackType(row["feedback_type"]),
            comment=row["sanitized_comment"],
            answer_snapshot=row["answer_snapshot"],
            verification_status=row["verification_status"],
            verification_decision=row["verification_decision"],
            route=row["route"],
            evidence_ids=ev_ids,
            tool_used=row["tool_used"],
            reviewed=bool(row["reviewed"]),
            review_status=ReviewStatus(row["review_status"]),
            improvement_candidate=cand,
            pii_scrubbed=bool(row["pii_scrubbed"]),
            reviewed_at=row["reviewed_at"],
            reviewer_notes=row["reviewer_notes"],
        )

    def get_feedback(self, feedback_id: str) -> Optional[FeedbackRecord]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM human_feedback WHERE feedback_id = ?", (feedback_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def list_feedback(
        self,
        status: Optional[ReviewStatus] = None,
        rating: Optional[int] = None,
        feedback_type: Optional[FeedbackType] = None,
        verification_status: Optional[str] = None,
        limit: int = 100,
    ) -> List[FeedbackRecord]:
        clauses = []
        params: List[Any] = []

        if status is not None:
            clauses.append("review_status = ?")
            params.append(status.value)
        if rating is not None:
            clauses.append("rating = ?")
            params.append(int(rating))
        if feedback_type is not None:
            clauses.append("feedback_type = ?")
            params.append(feedback_type.value)
        if verification_status is not None:
            clauses.append("verification_status = ?")
            params.append(verification_status.strip().upper())

        where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM human_feedback {where_str} ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]

    def update_review_status(
        self,
        feedback_id: str,
        review_status: ReviewStatus,
        reviewer_notes: Optional[str] = None,
    ) -> Optional[FeedbackRecord]:
        clean_notes = mask_pii(reviewer_notes) if reviewer_notes else None
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE human_feedback
                SET review_status = ?, reviewed = 1, reviewed_at = ?, reviewer_notes = ?
                WHERE feedback_id = ?
                """,
                (review_status.value, now_iso, clean_notes, feedback_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None
        return self.get_feedback(feedback_id)

    def clear_store(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM human_feedback")
            conn.commit()

    def verify_no_raw_pii(self, sensitive_strings: Optional[List[str]] = None) -> bool:
        if not sensitive_strings:
            return True
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT sanitized_comment, answer_snapshot, reviewer_notes FROM human_feedback")
            rows = cursor.fetchall()
            for row in rows:
                dump = f"{row['sanitized_comment'] or ''} {row['answer_snapshot'] or ''} {row['reviewer_notes'] or ''}"
                for s in sensitive_strings:
                    if s and s in dump:
                        return False
        return True


default_feedback_store = FeedbackStore()


def process_feedback_submission(
    submission: FeedbackSubmission,
    store: Optional[FeedbackStore] = None,
    trace_context: Optional[Dict[str, Any]] = None,
) -> FeedbackRecord:
    sanitized_comment = mask_pii(submission.comment) if submission.comment else None
    ctx = trace_context or lookup_trace_context(submission.trace_id)

    v_dec = ctx.get("verification_decision") or "UNKNOWN"
    v_stat = ctx.get("verification_status") or v_dec
    route = ctx.get("route")
    tool_used = ctx.get("tool_used")
    evidence_ids = list(ctx.get("evidence_ids") or [])
    answer_snapshot = ctx.get("answer_snapshot")
    if (v_dec or "").upper() == "REJECT" or (v_stat or "").upper() == "REJECT":
        answer_snapshot = FALLBACK_RESPONSE
    elif answer_snapshot:
        answer_snapshot = mask_pii(answer_snapshot)

    candidate = generate_improvement_candidate(
        trace_id=submission.trace_id,
        rating=submission.rating,
        feedback_type=submission.feedback_type,
        verification_decision=v_dec,
    )

    record = FeedbackRecord(
        trace_id=submission.trace_id,
        query_id=submission.query_id or ctx.get("query_id"),
        rating=submission.rating,
        feedback_type=submission.feedback_type,
        comment=sanitized_comment,
        answer_snapshot=answer_snapshot,
        verification_status=v_stat,
        verification_decision=v_dec,
        route=route,
        evidence_ids=evidence_ids,
        tool_used=tool_used,
        reviewed=False,
        review_status=ReviewStatus.PENDING,
        improvement_candidate=candidate,
        pii_scrubbed=True,
    )

    active_store = store or default_feedback_store
    return active_store.save_feedback(record)
