import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.reranker import compute_token_coverage, tokenize_text

GATE_PASS = "PASS"
GATE_RECOVER = "RECOVER"
GATE_FALLBACK = "FALLBACK"

DEFAULT_MIN_SIMILARITY_FLOOR = 0.35
DEFAULT_STRONG_SIMILARITY_THRESHOLD = 0.42
DEFAULT_STRONG_COVERAGE_THRESHOLD = 0.30
DEFAULT_STRONG_RERANK_THRESHOLD = 0.48
DEFAULT_RECOVERY_COVERAGE_THRESHOLD = 0.25


def extract_gate_signals(
    query_text: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not candidates or not isinstance(candidates, list):
        return {
            "s_sem": 0.0,
            "s_cov": 0.0,
            "s_rerank": 0.0,
            "s_topic": 0.0,
            "candidate_margin": 0.0,
            "same_source_consensus": False,
            "candidate_count": 0,
            "top_source": None,
            "rank2_source": None,
        }

    top1 = candidates[0] if isinstance(candidates[0], dict) else {}
    s_sem = round(max(0.0, min(1.0, max((float(c.get("similarity", 0.0)) for c in candidates if isinstance(c, dict)), default=float(top1.get("similarity", 0.0))))), 4)
    s_cov = round(max(0.0, min(1.0, float(top1.get("coverage_score", 0.0)))), 4)
    s_rerank = round(max(0.0, min(1.0, float(top1.get("rerank_score", 0.0)))), 4)
    s_topic = round(max(0.0, min(1.0, float(top1.get("topic_affinity", 0.0)))), 4)

    meta1 = top1.get("metadata", {}) or {}
    src1 = meta1.get("source") if isinstance(meta1, dict) else None

    if len(candidates) > 1 and isinstance(candidates[1], dict):
        top2 = candidates[1]
        rerank2 = float(top2.get("rerank_score", 0.0))
        margin = round(s_rerank - rerank2, 4)
        meta2 = top2.get("metadata", {}) or {}
        src2 = meta2.get("source") if isinstance(meta2, dict) else None
        same_source = (src1 is not None and src2 is not None and src1 == src2)
    else:
        margin = s_rerank
        src2 = None
        same_source = False

    return {
        "s_sem": s_sem,
        "s_cov": s_cov,
        "s_rerank": s_rerank,
        "s_topic": s_topic,
        "candidate_margin": margin,
        "same_source_consensus": same_source,
        "candidate_count": len(candidates),
        "top_source": src1,
        "rank2_source": src2,
    }


def evaluate_knowledge_gate(
    query_text: str,
    candidates: List[Dict[str, Any]],
    original_query: Optional[str] = None,
    min_similarity_floor: float = DEFAULT_MIN_SIMILARITY_FLOOR,
    strong_similarity_threshold: float = DEFAULT_STRONG_SIMILARITY_THRESHOLD,
    strong_coverage_threshold: float = DEFAULT_STRONG_COVERAGE_THRESHOLD,
    strong_rerank_threshold: float = DEFAULT_STRONG_RERANK_THRESHOLD,
    recovery_coverage_threshold: float = DEFAULT_RECOVERY_COVERAGE_THRESHOLD,
) -> Dict[str, Any]:
    if not candidates or not isinstance(candidates, list):
        return {
            "decision": GATE_FALLBACK,
            "is_passed": False,
            "confidence": 0.0,
            "signals": extract_gate_signals(query_text, []),
            "reason": "empty_candidate_pool",
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_check": None,
            "selected_candidates": [],
        }

    top1 = candidates[0]
    if not isinstance(top1, dict):
        return {
            "decision": GATE_FALLBACK,
            "is_passed": False,
            "confidence": 0.0,
            "signals": extract_gate_signals(query_text, []),
            "reason": "malformed_top_candidate",
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_check": None,
            "selected_candidates": [],
        }

    doc_text = str(top1.get("text", "")).strip()
    if not doc_text:
        return {
            "decision": GATE_FALLBACK,
            "is_passed": False,
            "confidence": 0.0,
            "signals": extract_gate_signals(query_text, candidates),
            "reason": "empty_candidate_text",
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_check": None,
            "selected_candidates": [],
        }

    signals = extract_gate_signals(query_text, candidates)
    s_sem = signals["s_sem"]
    s_cov = signals["s_cov"]
    s_rerank = signals["s_rerank"]

    if s_sem < min_similarity_floor:
        return {
            "decision": GATE_FALLBACK,
            "is_passed": False,
            "confidence": s_sem,
            "signals": signals,
            "reason": "semantic_similarity_below_floor",
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_check": None,
            "selected_candidates": [],
        }

    if (
        s_sem >= strong_similarity_threshold
        and s_cov >= strong_coverage_threshold
        and s_rerank >= strong_rerank_threshold
    ):
        return {
            "decision": GATE_PASS,
            "is_passed": True,
            "confidence": s_sem,
            "signals": signals,
            "reason": "sufficient_grounded_evidence",
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_check": None,
            "selected_candidates": candidates,
        }

    recovery_attempted = True
    recovery_successful = False
    recovery_check = None

    if original_query and isinstance(original_query, str):
        clean_orig = original_query.strip()
        clean_curr = query_text.strip()
        if clean_orig and clean_orig.lower() != clean_curr.lower():
            orig_tokens = [w for w in tokenize_text(clean_orig) if not w.endswith("_redacted") and not w.isdigit()]
            doc_tokens = set(tokenize_text(doc_text))
            orig_cov = compute_token_coverage(orig_tokens, doc_tokens)
            if orig_cov >= recovery_coverage_threshold and s_sem >= min_similarity_floor:
                recovery_successful = True
                recovery_check = "original_query_cross_check"
                signals["s_cov_original"] = orig_cov
                return {
                    "decision": GATE_PASS,
                    "is_passed": True,
                    "confidence": s_sem,
                    "signals": signals,
                    "reason": "recovered_via_original_query",
                    "recovery_attempted": recovery_attempted,
                    "recovery_successful": recovery_successful,
                    "recovery_check": recovery_check,
                    "selected_candidates": candidates,
                }

    if len(candidates) > 1 and isinstance(candidates[1], dict) and signals["same_source_consensus"]:
        top2 = candidates[1]
        sem2 = float(top2.get("similarity", 0.0))
        cov2 = float(top2.get("coverage_score", 0.0))
        rerank2 = float(top2.get("rerank_score", 0.0))
        if (
            max(s_cov, cov2) >= recovery_coverage_threshold
            and rerank2 >= strong_rerank_threshold
            and sem2 >= min_similarity_floor
        ):
            recovery_successful = True
            recovery_check = "rank2_consensus_check"
            return {
                "decision": GATE_PASS,
                "is_passed": True,
                "confidence": s_sem,
                "signals": signals,
                "reason": "recovered_via_rank2_consensus",
                "recovery_attempted": recovery_attempted,
                "recovery_successful": recovery_successful,
                "recovery_check": recovery_check,
                "selected_candidates": candidates,
            }

    return {
        "decision": GATE_FALLBACK,
        "is_passed": False,
        "confidence": s_sem,
        "signals": signals,
        "reason": "recovery_exhausted_insufficient_evidence",
        "recovery_attempted": recovery_attempted,
        "recovery_successful": False,
        "recovery_check": None,
        "selected_candidates": [],
    }


def safe_evaluate_knowledge_gate(
    query_text: str,
    candidates: List[Dict[str, Any]],
    original_query: Optional[str] = None,
    min_similarity_floor: float = DEFAULT_MIN_SIMILARITY_FLOOR,
    strong_similarity_threshold: float = DEFAULT_STRONG_SIMILARITY_THRESHOLD,
    strong_coverage_threshold: float = DEFAULT_STRONG_COVERAGE_THRESHOLD,
    strong_rerank_threshold: float = DEFAULT_STRONG_RERANK_THRESHOLD,
    recovery_coverage_threshold: float = DEFAULT_RECOVERY_COVERAGE_THRESHOLD,
) -> Dict[str, Any]:
    try:
        return evaluate_knowledge_gate(
            query_text=query_text,
            candidates=candidates,
            original_query=original_query,
            min_similarity_floor=min_similarity_floor,
            strong_similarity_threshold=strong_similarity_threshold,
            strong_coverage_threshold=strong_coverage_threshold,
            strong_rerank_threshold=strong_rerank_threshold,
            recovery_coverage_threshold=recovery_coverage_threshold,
        )
    except Exception:
        return {
            "decision": GATE_FALLBACK,
            "is_passed": False,
            "confidence": 0.0,
            "signals": extract_gate_signals(query_text, []),
            "reason": "internal_evaluation_exception",
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_check": None,
            "selected_candidates": [],
        }
