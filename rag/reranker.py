import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_RERANK_TOP_K = 3
DEFAULT_W_SEM = 0.45
DEFAULT_W_COV = 0.25
DEFAULT_W_TOPIC = 0.10
DEFAULT_W_RRF = 0.20

ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with", "what", "how", "when",
    "where", "who", "which", "why", "can", "could", "would", "should",
    "do", "does", "did", "i", "you", "my", "your", "we", "our", "me",
    "them", "this", "these", "those", "have", "had", "if", "but", "so"
}

CONVERSATIONAL_STOPWORDS: Set[str] = {
    "hi", "hello", "hey", "greetings", "namaste", "morning", "afternoon", "evening",
    "please", "thanks", "thank", "good", "help", "tell", "know", "assist",
}
ENGLISH_STOPWORDS.update(CONVERSATIONAL_STOPWORDS)

TOKEN_PATTERN = re.compile(r"\b[a-z0-9_]+\b")


def tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    words = TOKEN_PATTERN.findall(text.lower())
    return [w for w in words if w not in ENGLISH_STOPWORDS and len(w) > 1]


def compute_token_coverage(query_tokens: List[str], doc_tokens: Set[str]) -> float:
    if not query_tokens:
        return 0.0
    matched = sum(1 for token in query_tokens if token in doc_tokens)
    return round(matched / float(len(query_tokens)), 4)


def compute_topic_affinity(query_tokens: List[str], source_filename: Optional[str]) -> float:
    if not query_tokens or not source_filename:
        return 0.0
    cleaned_source = source_filename.lower().replace(".md", "").replace("_", " ")
    source_tokens = set(TOKEN_PATTERN.findall(cleaned_source)) - ENGLISH_STOPWORDS
    matched = sum(1 for token in query_tokens if token in source_tokens)
    return 1.0 if matched > 0 else 0.0


def rerank_candidates(
    query_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = DEFAULT_RERANK_TOP_K,
    w_sem: float = DEFAULT_W_SEM,
    w_cov: float = DEFAULT_W_COV,
    w_topic: float = DEFAULT_W_TOPIC,
    w_rrf: float = DEFAULT_W_RRF,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    if len(candidates) == 1:
        single = candidates[0].copy()
        single["rerank_score"] = 1.0
        return [single]

    q_tokens = tokenize_text(query_text)
    max_rrf = max((float(c.get("rrf_score", 0.0)) for c in candidates), default=1.0)
    if max_rrf <= 0.0:
        max_rrf = 1.0

    scored_candidates = []
    for cand in candidates:
        sim = max(0.0, min(1.0, float(cand.get("similarity", 0.0))))
        doc_text = str(cand.get("text", ""))
        doc_tokens = set(tokenize_text(doc_text))
        coverage = compute_token_coverage(q_tokens, doc_tokens)

        meta = cand.get("metadata", {}) or {}
        source = meta.get("source", "")
        topic_affinity = compute_topic_affinity(q_tokens, source)

        norm_rrf = max(0.0, float(cand.get("rrf_score", 0.0))) / max_rrf

        rerank_score = (
            w_sem * sim
            + w_cov * coverage
            + w_topic * topic_affinity
            + w_rrf * norm_rrf
        )

        item = cand.copy()
        item["rerank_score"] = round(rerank_score, 4)
        item["coverage_score"] = coverage
        item["topic_affinity"] = topic_affinity
        scored_candidates.append(item)

    scored_candidates.sort(
        key=lambda x: (
            -round(x["rerank_score"], 2),
            int((x.get("metadata") or {}).get("chunk_index", 0)),
            -x["rerank_score"],
            -round(float(x.get("similarity", 0.0)), 4),
            -round(float(x.get("rrf_score", 0.0)), 6),
            str(x.get("id", "")),
        )
    )

    return scored_candidates[:top_k]


def safe_rerank_candidates(
    query_text: str,
    candidates: List[Dict[str, Any]],
    top_k: int = DEFAULT_RERANK_TOP_K,
    w_sem: float = DEFAULT_W_SEM,
    w_cov: float = DEFAULT_W_COV,
    w_topic: float = DEFAULT_W_TOPIC,
    w_rrf: float = DEFAULT_W_RRF,
) -> List[Dict[str, Any]]:
    try:
        return rerank_candidates(
            query_text=query_text,
            candidates=candidates,
            top_k=top_k,
            w_sem=w_sem,
            w_cov=w_cov,
            w_topic=w_topic,
            w_rrf=w_rrf,
        )
    except Exception:
        return candidates[:top_k] if candidates else []
