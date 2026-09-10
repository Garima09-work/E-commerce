import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.reranker import ENGLISH_STOPWORDS, TOKEN_PATTERN, tokenize_text

GENERIC_DOMAIN_TERMS: Set[str] = {
    "return", "returns", "order", "orders", "product", "products",
    "item", "items", "policy", "policies", "delivery", "deliveries",
    "customer", "customers", "nykaa", "service", "services",
    "rule", "rules", "information", "support", "question", "questions",
}

CATEGORY_KEYWORDS: Dict[str, Set[str]] = {
    "beauty": {
        "beauty", "cosmetic", "cosmetics", "skincare", "perfume",
        "perfumes", "fragrance", "fragrances", "lipstick", "foundation",
        "makeup", "lotion", "cream",
    },
    "apparel": {
        "apparel", "clothing", "dress", "footwear", "shoes", "innerwear",
        "intimate", "lingerie", "size", "fit", "exchange",
    },
    "electronics": {
        "electronics", "appliance", "appliances", "tools", "styling",
        "device", "devices", "warranty", "dryer", "straightener",
    },
}

NUMERICAL_TIME_PATTERN = re.compile(
    r"\b(\d+\s*(?:business\s+days?|working\s+days?|days?|hours?|weeks?|months?)|within\s+\d+|after\s+\d+|before\s+\d+)\b",
    re.IGNORECASE,
)

EXCLUSION_PATTERN = re.compile(
    r"\b(non-returnable|non-refundable|cannot|not\s+eligible|ineligible|except|unless|only|strictly|void|prohibited|mandatory|required|must|condition|conditions|terms|restrictions?|never)\b",
    re.IGNORECASE,
)

FINANCIAL_PATTERN = re.compile(
    r"\b(inr|rs\.?|rupees?|\d+%\s*|refund|refunds|amount|charge|charges|fee|fees|cost|costs|deducted|debited|reimburse|reimbursement)\b",
    re.IGNORECASE,
)

PROCEDURE_PATTERN = re.compile(
    r"\b(inspect|inspection|verification|portal|upload|tracking|sla|timeline|timelines|carrier|courier|pickup|delivery|unopened|seal|packaging|tags|applicator|receipt|invoice)\b",
    re.IGNORECASE,
)

LAST_COMPRESSION_RESULT: Dict[str, Any] = {}


def split_into_sentences(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    clean = text.strip()
    if not clean:
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [clean]


KB_CACHE: Dict[str, str] = {}


def get_kb_document_text(source_filename: Optional[str]) -> str:
    if not source_filename:
        return ""
    if source_filename in KB_CACHE:
        return KB_CACHE[source_filename]
    kb_path = ROOT_DIR / "knowledge_base" / source_filename
    if kb_path.exists():
        try:
            content = kb_path.read_text(encoding="utf-8")
            from rag.chunking import extract_clean_text
            clean = extract_clean_text(content) if "#" in content else content.strip()
            KB_CACHE[source_filename] = clean
            return clean
        except Exception:
            pass
    return ""


def heal_sentence_from_kb(sentence: str, source_filename: Optional[str]) -> str:
    if not sentence:
        return ""
    clean_s = sentence.strip()
    if not clean_s or not source_filename:
        return clean_s

    kb_text = get_kb_document_text(source_filename)
    if not kb_text:
        return clean_s

    kb_sentences = split_into_sentences(kb_text)
    for kb_s in kb_sentences:
        if clean_s == kb_s:
            return kb_s

    # If clean_s is an incomplete fragment from chunk boundaries
    for kb_s in kb_sentences:
        if clean_s in kb_s or clean_s.lower() in kb_s.lower():
            return kb_s

    return clean_s


def match_token_flexible(token_a: str, token_b: str) -> bool:
    if token_a == token_b:
        return True
    stem_a = token_a.rstrip("s")
    stem_b = token_b.rstrip("s")
    if stem_a and stem_a == stem_b:
        return True
    if len(token_a) >= 4 and len(token_b) >= 4:
        if token_a.startswith(token_b) or token_b.startswith(token_a):
            return True
    return False


def score_sentence_relevance(
    sentence: str,
    query_tokens: List[str],
    source_filename: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    if not sentence:
        return 0.0, {
            "overlap_count": 0,
            "has_constraint": False,
            "has_exclusion": False,
            "is_conflicting": False,
        }

    s_tokens = tokenize_text(sentence)
    if not s_tokens:
        return 0.0, {
            "overlap_count": 0,
            "has_constraint": False,
            "has_exclusion": False,
            "is_conflicting": False,
        }

    specific_query = [qt for qt in query_tokens if qt not in GENERIC_DOMAIN_TERMS]

    specific_overlap = sum(
        1 for qt in specific_query
        if any(match_token_flexible(qt, st) for st in s_tokens)
    )
    generic_overlap = sum(
        1 for qt in query_tokens
        if qt in GENERIC_DOMAIN_TERMS and any(match_token_flexible(qt, st) for st in s_tokens)
    )

    query_categories = set()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(match_token_flexible(qt, kw) for qt in query_tokens for kw in kws):
            query_categories.add(cat)

    sentence_categories = set()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(match_token_flexible(st, kw) for st in s_tokens for kw in kws):
            sentence_categories.add(cat)

    is_conflicting = bool(
        query_categories
        and sentence_categories
        and not (query_categories & sentence_categories)
    )
    has_category_match = bool(query_categories and (query_categories & sentence_categories))

    has_num_time = bool(NUMERICAL_TIME_PATTERN.search(sentence))
    has_exclusion = bool(EXCLUSION_PATTERN.search(sentence))
    has_financial = bool(FINANCIAL_PATTERN.search(sentence))
    has_procedure = bool(PROCEDURE_PATTERN.search(sentence))

    topic_tokens = set()
    if source_filename:
        clean_src = source_filename.lower().replace(".md", "").replace("_", " ")
        topic_tokens = set(tokenize_text(clean_src))

    topic_overlap = sum(
        1 for st in s_tokens
        if any(match_token_flexible(st, tt) for tt in topic_tokens)
    )

    score = specific_overlap * 2.0 + generic_overlap * 0.5
    if has_category_match:
        score += 1.0
    if has_exclusion:
        score += 0.50
    if has_num_time:
        score += 0.40
    if has_financial:
        score += 0.20
    if has_procedure:
        score += 0.15
    if topic_overlap > 0:
        score += 0.20
    if is_conflicting:
        score -= 3.0

    audit = {
        "overlap_count": specific_overlap + generic_overlap,
        "specific_overlap": specific_overlap,
        "generic_overlap": generic_overlap,
        "has_constraint": has_num_time or has_financial,
        "has_exclusion": has_exclusion,
        "has_procedure": has_procedure,
        "has_category_match": has_category_match,
        "is_conflicting": is_conflicting,
        "topic_overlap": topic_overlap,
        "score": round(score, 4),
    }
    return round(score, 4), audit


def compress_candidate_text(
    candidate_text: str,
    query_tokens: List[str],
    source_filename: Optional[str] = None,
) -> Tuple[str, bool, int, int]:
    if not candidate_text or not isinstance(candidate_text, str):
        return candidate_text, False, 0, 0

    raw_sentences = split_into_sentences(candidate_text)
    if not raw_sentences:
        return candidate_text, False, 0, 0

    # Heal any sentence fragments caused by chunk boundary slicing
    sentences = [heal_sentence_from_kb(s, source_filename) for s in raw_sentences]
    deduped_sentences = []
    for s in sentences:
        if not deduped_sentences or s != deduped_sentences[-1]:
            deduped_sentences.append(s)
    sentences = deduped_sentences

    if len(sentences) <= 1:
        return " ".join(sentences), False, len(raw_sentences), len(sentences)

    clean_q_tokens = [
        qt for qt in query_tokens
        if qt not in ENGLISH_STOPWORDS and len(qt) > 1
    ]

    scored_sentences = []
    for idx, s in enumerate(sentences):
        score, audit = score_sentence_relevance(s, clean_q_tokens, source_filename)
        scored_sentences.append((idx, s, score, audit))

    has_any_query_overlap = any(
        item[3]["overlap_count"] > 0 or item[3]["has_category_match"]
        for item in scored_sentences
    )

    if not has_any_query_overlap:
        return " ".join(sentences), False, len(sentences), len(sentences)

    specific_query = [qt for qt in clean_q_tokens if qt not in GENERIC_DOMAIN_TERMS]

    kept_indices = set()
    for idx, s, score, audit in scored_sentences:
        if audit["is_conflicting"]:
            continue

        keep = False
        if specific_query:
            if (
                audit["specific_overlap"] > 0
                or audit["has_category_match"]
                or (audit["has_exclusion"] and not audit["is_conflicting"])
                or (audit["has_constraint"] and not audit["is_conflicting"] and score >= 0.8)
            ):
                keep = True
        else:
            if score >= 0.50:
                keep = True

        if keep:
            kept_indices.add(idx)

    # Contextual completeness / Neighbor preservation rule:
    # 1. If an exclusion or restriction sentence is kept, preserve the preceding context / introductory rule
    #    so that restrictions are not presented without their baseline policy context.
    # 2. For broad policy queries (specific_query is empty), always ensure sentence 0 (the lead policy statement) is preserved.
    for idx, s, score, audit in scored_sentences:
        if idx in kept_indices and audit["has_exclusion"]:
            if idx > 0 and not scored_sentences[idx - 1][3]["is_conflicting"]:
                kept_indices.add(idx - 1)
            if not scored_sentences[0][3]["is_conflicting"]:
                kept_indices.add(0)

    if not specific_query and scored_sentences:
        if not scored_sentences[0][3]["is_conflicting"]:
            kept_indices.add(0)

    kept = [(idx, sentences[idx]) for idx in sorted(kept_indices)]

    if not kept:
        return " ".join(sentences), False, len(sentences), len(sentences)

    was_compressed = (len(kept) < len(sentences)) or (" ".join(item[1] for item in kept) != candidate_text)

    if len(kept) == len(sentences) and not was_compressed:
        return candidate_text, False, len(sentences), len(sentences)

    kept.sort(key=lambda x: x[0])
    compressed = " ".join(item[1] for item in kept)
    return compressed, was_compressed, len(sentences), len(kept)


def compress_candidates(
    query_text: str,
    candidates: List[Dict[str, Any]],
    original_query: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    global LAST_COMPRESSION_RESULT

    if not candidates or not isinstance(candidates, list):
        empty_meta = {
            "compression_applied": False,
            "original_chunk_count": 0,
            "compressed_chunk_count": 0,
            "original_sentence_count": 0,
            "compressed_sentence_count": 0,
            "removed_sentence_count": 0,
            "preserved_sentence_count": 0,
            "original_chars": 0,
            "compressed_chars": 0,
            "original_words": 0,
            "compressed_words": 0,
            "compression_ratio": 0.0,
            "source_ids": [],
            "chunk_ids": [],
        }
        LAST_COMPRESSION_RESULT = empty_meta
        return [], empty_meta

    if not query_text or not isinstance(query_text, str) or not query_text.strip():
        source_ids = []
        chunk_ids = []
        orig_chars = 0
        orig_words = 0
        for c in candidates:
            if isinstance(c, dict):
                txt = str(c.get("text", ""))
                orig_chars += len(txt)
                orig_words += len(txt.split())
                chunk_ids.append(str(c.get("id", "")))
                meta = c.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("source"):
                    source_ids.append(meta["source"])
        fallback_meta = {
            "compression_applied": False,
            "original_chunk_count": len(candidates),
            "compressed_chunk_count": len(candidates),
            "original_sentence_count": 0,
            "compressed_sentence_count": 0,
            "removed_sentence_count": 0,
            "preserved_sentence_count": 0,
            "original_chars": orig_chars,
            "compressed_chars": orig_chars,
            "original_words": orig_words,
            "compressed_words": orig_words,
            "compression_ratio": 0.0,
            "source_ids": list(dict.fromkeys(source_ids)),
            "chunk_ids": chunk_ids,
        }
        LAST_COMPRESSION_RESULT = fallback_meta
        return [c.copy() if isinstance(c, dict) else c for c in candidates], fallback_meta

    query_tokens = tokenize_text(query_text)
    if original_query and isinstance(original_query, str):
        clean_orig = original_query.strip()
        if clean_orig and clean_orig.lower() != query_text.strip().lower():
            orig_tokens = tokenize_text(clean_orig)
            for ot in orig_tokens:
                if ot not in query_tokens:
                    query_tokens.append(ot)

    compressed_candidates = []
    any_applied = False
    total_orig_sentences = 0
    total_comp_sentences = 0
    total_orig_chars = 0
    total_comp_chars = 0
    total_orig_words = 0
    total_comp_words = 0
    source_ids = []
    chunk_ids = []

    for c in candidates:
        if not isinstance(c, dict):
            compressed_candidates.append(c)
            continue

        raw_text = str(c.get("text", ""))
        chunk_id = str(c.get("id", ""))
        chunk_ids.append(chunk_id)

        meta = c.get("metadata") or {}
        src_name = meta.get("source") if isinstance(meta, dict) else None
        if src_name:
            source_ids.append(src_name)

        orig_len = len(raw_text)
        orig_w = len(raw_text.split())
        total_orig_chars += orig_len
        total_orig_words += orig_w

        comp_text, was_applied, n_orig, n_comp = compress_candidate_text(
            candidate_text=raw_text,
            query_tokens=query_tokens,
            source_filename=src_name,
        )

        total_orig_sentences += n_orig
        total_comp_sentences += n_comp
        total_comp_chars += len(comp_text)
        total_comp_words += len(comp_text.split())

        if was_applied:
            any_applied = True

        comp_item = c.copy()
        comp_item["original_text"] = raw_text
        comp_item["text"] = comp_text
        comp_item["compression_applied"] = was_applied
        comp_item["original_sentence_count"] = n_orig
        comp_item["compressed_sentence_count"] = n_comp
        comp_item["removed_sentence_count"] = max(0, n_orig - n_comp)
        compressed_candidates.append(comp_item)

    removed_sentences = max(0, total_orig_sentences - total_comp_sentences)
    ratio = (
        round(1.0 - (float(total_comp_chars) / float(total_orig_chars)), 4)
        if total_orig_chars > 0
        else 0.0
    )

    metadata = {
        "compression_applied": any_applied,
        "original_chunk_count": len(candidates),
        "compressed_chunk_count": len(compressed_candidates),
        "original_sentence_count": total_orig_sentences,
        "compressed_sentence_count": total_comp_sentences,
        "removed_sentence_count": removed_sentences,
        "preserved_sentence_count": total_comp_sentences,
        "original_chars": total_orig_chars,
        "compressed_chars": total_comp_chars,
        "original_words": total_orig_words,
        "compressed_words": total_comp_words,
        "compression_ratio": max(0.0, ratio),
        "source_ids": list(dict.fromkeys(source_ids)),
        "chunk_ids": chunk_ids,
    }

    LAST_COMPRESSION_RESULT = metadata
    return compressed_candidates, metadata


def safe_compress_candidates(
    query_text: str,
    candidates: List[Dict[str, Any]],
    original_query: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        return compress_candidates(
            query_text=query_text,
            candidates=candidates,
            original_query=original_query,
        )
    except Exception:
        fallback_candidates = []
        source_ids = []
        chunk_ids = []
        orig_chars = 0
        orig_words = 0
        for c in candidates if isinstance(candidates, list) else []:
            if isinstance(c, dict):
                item = c.copy()
                txt = str(item.get("text", ""))
                orig_chars += len(txt)
                orig_words += len(txt.split())
                chunk_ids.append(str(item.get("id", "")))
                meta = item.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("source"):
                    source_ids.append(meta["source"])
                item["original_text"] = txt
                item["compression_applied"] = False
                fallback_candidates.append(item)
            else:
                fallback_candidates.append(c)

        fallback_meta = {
            "compression_applied": False,
            "original_chunk_count": len(candidates) if isinstance(candidates, list) else 0,
            "compressed_chunk_count": len(fallback_candidates),
            "original_sentence_count": 0,
            "compressed_sentence_count": 0,
            "removed_sentence_count": 0,
            "preserved_sentence_count": 0,
            "original_chars": orig_chars,
            "compressed_chars": orig_chars,
            "original_words": orig_words,
            "compressed_words": orig_words,
            "compression_ratio": 0.0,
            "source_ids": list(dict.fromkeys(source_ids)),
            "chunk_ids": chunk_ids,
            "fallback_error": True,
        }
        return fallback_candidates, fallback_meta


def get_last_compression_result() -> Dict[str, Any]:
    return dict(LAST_COMPRESSION_RESULT)
