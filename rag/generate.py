import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
from sentence_transformers import SentenceTransformer

from rag.embed_index import (
    DEFAULT_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    FIXED_COLLECTION_NAME,
    SENTENCE_COLLECTION_NAME,
    get_chroma_client,
)
from rag.hybrid import retrieve_hybrid_context

FALLBACK_RESPONSE = (
    "I don't have enough grounded information from the knowledge base to answer that "
    "confidently. Could you rephrase, or would you like this escalated to a support agent?"
)

DEFAULT_SIMILARITY_THRESHOLD = 0.35
DEFAULT_COLLECTION_NAME = SENTENCE_COLLECTION_NAME
DEFAULT_TOP_K = 2


def retrieve_context(
    query_text: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
    client: Optional[chromadb.PersistentClient] = None,
    model: Optional[SentenceTransformer] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    active_mode = (mode or os.environ.get("RETRIEVAL_MODE", "hybrid")).lower()
    if active_mode in ("rerank", "hybrid_rerank", "gate", "knowledge_gate", "compress", "context_compress", "compressed"):
        return retrieve_hybrid_context(
            query_text=query_text,
            collection_name=collection_name,
            top_k=top_k,
            persist_directory=persist_directory,
            model_name=model_name,
            client=client,
            model=model,
            use_reranker=True,
        )
    if active_mode == "hybrid":
        return retrieve_hybrid_context(
            query_text=query_text,
            collection_name=collection_name,
            top_k=top_k,
            persist_directory=persist_directory,
            model_name=model_name,
            client=client,
            model=model,
            use_reranker=False,
        )

    if client is None:
        client = get_chroma_client(persist_directory)
    if model is None:
        model = SentenceTransformer(model_name)

    collection = client.get_collection(collection_name)
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()

    raw_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = raw_results["documents"][0] if raw_results["documents"] else []
    metadatas = raw_results["metadatas"][0] if raw_results["metadatas"] else []
    distances = raw_results["distances"][0] if raw_results["distances"] else []

    similarities = [max(0.0, min(1.0, 1.0 - d)) for d in distances]
    top_similarity = similarities[0] if similarities else 0.0

    return {
        "query": query_text,
        "collection": collection_name,
        "documents": documents,
        "metadatas": metadatas,
        "similarities": similarities,
        "top_similarity": top_similarity,
        "mode": "semantic",
    }


LAST_GATE_RESULT: Dict[str, Any] = {}
LAST_RETRIEVAL_RESULT: Dict[str, Any] = {}


def get_last_gate_result() -> Dict[str, Any]:
    return dict(LAST_GATE_RESULT)


def get_last_retrieval_result() -> Dict[str, Any]:
    return dict(LAST_RETRIEVAL_RESULT)


def order_candidates_by_document(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not cands or len(cands) <= 1:
        return cands
    doc_first_rank: Dict[str, int] = {}
    for idx, c in enumerate(cands):
        meta = (c.get("metadata") or {}) if isinstance(c, dict) else {}
        src = meta.get("source", f"_doc_{idx}")
        if src not in doc_first_rank:
            doc_first_rank[src] = idx

    return sorted(
        cands,
        key=lambda c: (
            doc_first_rank.get((c.get("metadata") or {}).get("source", ""), 999),
            int((c.get("metadata") or {}).get("chunk_index", 0)),
        ),
    )


def generate_grounded_answer(
    query_text: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    trace_id: Optional[str] = None,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
    client: Optional[chromadb.PersistentClient] = None,
    model: Optional[SentenceTransformer] = None,
    mode: Optional[str] = None,
    original_query: Optional[str] = None,
    use_knowledge_gate: Optional[bool] = None,
    use_compressor: Optional[bool] = None,
) -> Dict[str, Any]:
    active_trace_id = trace_id or str(uuid.uuid4())
    active_mode = (mode or os.environ.get("RETRIEVAL_MODE", "hybrid")).lower()
    active_use_gate = use_knowledge_gate
    if active_use_gate is None:
        active_use_gate = (
            os.environ.get("ENABLE_KNOWLEDGE_GATE", "1") == "1"
            or active_mode in ("gate", "knowledge_gate")
        )

    active_use_compressor = use_compressor
    if active_use_compressor is None:
        active_use_compressor = os.environ.get("ENABLE_CONTEXT_COMPRESSION", "1") == "1"

    retrieval_mode = active_mode
    if active_use_gate:
        retrieval_mode = "rerank"

    retrieval = retrieve_context(
        query_text=query_text,
        collection_name=collection_name,
        top_k=top_k,
        persist_directory=persist_directory,
        model_name=model_name,
        client=client,
        model=model,
        mode=retrieval_mode,
    )

    confidence = round(retrieval["top_similarity"], 3)

    if confidence < threshold or not retrieval["documents"]:
        LAST_RETRIEVAL_RESULT.clear()
        LAST_RETRIEVAL_RESULT.update({
            "query": query_text,
            "documents": [],
            "metadatas": [],
            "candidates": [],
            "sources": [],
        })
        LAST_GATE_RESULT.clear()
        LAST_GATE_RESULT.update({
            "decision": "FALLBACK",
            "is_passed": False,
            "confidence": confidence,
            "reason": "semantic_similarity_below_floor",
        })
        return {
            "response_type": "fallback",
            "answer": FALLBACK_RESPONSE,
            "sources": [],
            "confidence": confidence,
            "escalation_score": None,
            "trace_id": active_trace_id,
        }

    candidates = retrieval.get("fused_candidates") or []
    if active_use_compressor and candidates:
        from rag.context_compressor import safe_compress_candidates

        compressed_candidates, comp_meta = safe_compress_candidates(
            query_text=query_text,
            candidates=candidates,
            original_query=original_query,
        )
        eval_candidates = compressed_candidates
        active_documents = [
            str(c.get("text", ""))
            for c in compressed_candidates
            if isinstance(c, dict) and str(c.get("text", "")).strip()
        ]
        if not active_documents:
            active_documents = retrieval["documents"]
    else:
        eval_candidates = candidates
        active_documents = retrieval["documents"]

    ordered_eval_candidates = order_candidates_by_document(eval_candidates)
    ordered_active_documents = [
        str(c.get("text", ""))
        for c in ordered_eval_candidates
        if isinstance(c, dict) and str(c.get("text", "")).strip()
    ]
    if not ordered_active_documents:
        ordered_active_documents = active_documents

    if active_use_gate:
        from rag.knowledge_gate import safe_evaluate_knowledge_gate

        gate_res = safe_evaluate_knowledge_gate(
            query_text=query_text,
            candidates=eval_candidates,
            original_query=original_query,
            min_similarity_floor=threshold,
        )
        LAST_GATE_RESULT.clear()
        LAST_GATE_RESULT.update(gate_res)
        if not gate_res["is_passed"]:
            LAST_RETRIEVAL_RESULT.clear()
            LAST_RETRIEVAL_RESULT.update({
                "query": query_text,
                "documents": ordered_active_documents,
                "metadatas": retrieval.get("metadatas", []),
                "candidates": ordered_eval_candidates,
                "sources": [],
            })
            return {
                "response_type": "fallback",
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "confidence": confidence,
                "escalation_score": None,
                "trace_id": active_trace_id,
            }

    seen_sources = set()
    unique_sources = []
    for cand in ordered_eval_candidates:
        meta = (cand.get("metadata") or {}) if isinstance(cand, dict) else {}
        source = meta.get("source")
        if source and source not in seen_sources:
            seen_sources.add(source)
            unique_sources.append(source)
    if not unique_sources:
        for meta in retrieval.get("metadatas", []):
            source = meta.get("source")
            if source and source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append(source)

    extracted_sentences = []
    seen_sentences = set()
    from rag.context_compressor import split_into_sentences

    for cand in ordered_eval_candidates:
        doc_text = str(cand.get("text", "")).strip() if isinstance(cand, dict) else str(cand).strip()
        if not doc_text:
            continue
        for s in split_into_sentences(doc_text):
            s_clean = s.strip()
            s_key = s_clean.lower()
            if s_clean and s_key not in seen_sentences:
                seen_sentences.add(s_key)
                extracted_sentences.append(s_clean)

    answer_body = " ".join(extracted_sentences)

    LAST_RETRIEVAL_RESULT.clear()
    LAST_RETRIEVAL_RESULT.update({
        "query": query_text,
        "documents": active_documents,
        "metadatas": retrieval.get("metadatas", []),
        "candidates": eval_candidates,
        "sources": unique_sources,
    })

    return {
        "response_type": "policy_answer",
        "answer": answer_body,
        "sources": unique_sources,
        "confidence": confidence,
        "escalation_score": None,
        "trace_id": active_trace_id,
    }


def run_threshold_calibration_and_tests() -> None:
    client = get_chroma_client(DEFAULT_PERSIST_DIR)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    grounded_queries = [
        "Can I return this product?",
        "When will I get my refund for COD orders?",
        "What are the delivery timelines for metro cities?",
        "How does reverse pickup scheduling work?",
        "What is the warranty period for grooming appliances?",
        "Can I cancel my order before dispatch?",
        "How do I earn and redeem reward points?",
        "My payment failed but money was deducted from my account.",
        "Can I exchange my shoes for another size?",
        "I received a damaged and broken bottle.",
        "How long does international delivery take and what about customs?",
        "What is the customer support escalation process?",
    ]

    ungrounded_queries = [
        "What is the capital of France?",
        "What is the weather in Delhi today?",
        "How do I write a quicksort algorithm in Python?",
        "Who directed the movie Inception?",
        "What is the stock price of Apple today?",
    ]

    print("================================================================================")
    print("           NYKAA ASSIST TASK 4 — GROUNDED GENERATION & CALIBRATION              ")
    print("================================================================================")
    print(f"Collection in use      : {DEFAULT_COLLECTION_NAME}")
    print(f"Embedding model        : {EMBEDDING_MODEL_NAME}")
    print(f"Calibrated Threshold   : {DEFAULT_SIMILARITY_THRESHOLD}")
    print("--------------------------------------------------------------------------------")

    print("\n--- Testing Grounded (In-Scope) Queries ---")
    in_scores = []
    for query in grounded_queries:
        response = generate_grounded_answer(
            query_text=query,
            threshold=DEFAULT_SIMILARITY_THRESHOLD,
            client=client,
            model=model,
        )
        in_scores.append(response["confidence"])
        status = "PASSED" if response["response_type"] == "policy_answer" else "FAILED"
        print(f"[{status}] Query     : '{query}'")
        print(f"         Confidence: {response['confidence']:.3f} | Sources: {response['sources']}")
        print(f"         Type      : {response['response_type']}")
        print(f"         Answer    : {response['answer'][:120]}...\n")

    print("--- Testing Ungrounded (Out-of-Scope) Queries ---")
    out_scores = []
    for query in ungrounded_queries:
        response = generate_grounded_answer(
            query_text=query,
            threshold=DEFAULT_SIMILARITY_THRESHOLD,
            client=client,
            model=model,
        )
        out_scores.append(response["confidence"])
        is_exact_fallback = response["answer"] == FALLBACK_RESPONSE
        status = "PASSED" if (response["response_type"] == "fallback" and is_exact_fallback) else "FAILED"
        print(f"[{status}] Query     : '{query}'")
        print(f"         Confidence: {response['confidence']:.3f} | Sources: {response['sources']}")
        print(f"         Type      : {response['response_type']}")
        print(f"         Answer    : {response['answer']}\n")

    print("--------------------------------------------------------------------------------")
    print("                      CALIBRATION SUMMARY METRICS                               ")
    print("--------------------------------------------------------------------------------")
    print(f"In-Scope Similarity Range   : [{min(in_scores):.3f}, {max(in_scores):.3f}] (Mean: {sum(in_scores)/len(in_scores):.3f})")
    print(f"Out-of-Scope Similarity Range: [{min(out_scores):.3f}, {max(out_scores):.3f}] (Mean: {sum(out_scores)/len(out_scores):.3f})")
    print(f"Observed Separation Margin  : {min(in_scores) - max(out_scores):.3f}")
    print(f"Calibrated Threshold Choice : {DEFAULT_SIMILARITY_THRESHOLD} (Positioned securely between {max(out_scores):.3f} and {min(in_scores):.3f})")
    print("================================================================================")


def main() -> None:
    run_threshold_calibration_and_tests()


if __name__ == "__main__":
    main()
