import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.embed_index import (
    DEFAULT_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    FIXED_COLLECTION_NAME,
    get_chroma_client,
)
from rag.lexical import BM25Index, get_bm25_index

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_DEPTH = 10
_EMBEDDING_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def get_cached_embeddings(
    collection: chromadb.Collection,
    collection_name: str,
) -> Dict[str, np.ndarray]:
    if collection_name in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[collection_name]
    raw = collection.get(include=["embeddings"])
    ids = raw.get("ids", [])
    embeddings = raw.get("embeddings", [])
    mapping = {cid: np.array(emb, dtype=np.float32) for cid, emb in zip(ids, embeddings)}
    _EMBEDDING_CACHE[collection_name] = mapping
    return mapping


def clear_embedding_cache() -> None:
    _EMBEDDING_CACHE.clear()


def reciprocal_rank_fusion(
    semantic_candidates: List[Dict[str, Any]],
    lexical_candidates: List[Dict[str, Any]],
    k_rrf: int = DEFAULT_RRF_K,
    w_sem: float = 1.0,
    w_lex: float = 1.0,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    rrf_scores: Dict[str, float] = {}
    doc_registry: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(semantic_candidates):
        doc_id = item["id"]
        doc_registry[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
            w_sem / float(k_rrf + rank + 1)
        )

    for rank, item in enumerate(lexical_candidates):
        doc_id = item["id"]
        if doc_id not in doc_registry:
            doc_registry[doc_id] = item
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
            w_lex / float(k_rrf + rank + 1)
        )

    sorted_doc_ids = sorted(
        rrf_scores.keys(),
        key=lambda did: (
            -rrf_scores[did],
            -doc_registry[did].get("similarity", 0.0),
            did,
        ),
    )

    fused_results = []
    for doc_id in sorted_doc_ids[:top_k]:
        candidate_item = doc_registry[doc_id].copy()
        candidate_item["rrf_score"] = round(rrf_scores[doc_id], 6)
        fused_results.append(candidate_item)

    return fused_results


def retrieve_hybrid_context(
    query_text: str,
    collection_name: str = FIXED_COLLECTION_NAME,
    top_k: int = 3,
    candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
    k_rrf: int = DEFAULT_RRF_K,
    w_sem: float = 1.0,
    w_lex: float = 1.0,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
    client: Optional[chromadb.PersistentClient] = None,
    model: Optional[SentenceTransformer] = None,
    use_reranker: bool = False,
    use_compressor: bool = False,
) -> Dict[str, Any]:
    active_client = client or get_chroma_client(persist_directory)
    active_model = model or SentenceTransformer(model_name)
    collection = active_client.get_collection(collection_name)

    query_embedding = active_model.encode(query_text, normalize_embeddings=True)
    query_vector = query_embedding.tolist()

    total_chunks = collection.count()
    actual_depth = min(candidate_depth, total_chunks) if total_chunks > 0 else 0

    semantic_candidates: List[Dict[str, Any]] = []
    if actual_depth > 0:
        raw_sem = collection.query(
            query_embeddings=[query_vector],
            n_results=actual_depth,
            include=["documents", "metadatas", "distances"],
        )
        sem_ids = raw_sem["ids"][0] if raw_sem["ids"] else []
        sem_docs = raw_sem["documents"][0] if raw_sem["documents"] else []
        sem_metas = raw_sem["metadatas"][0] if raw_sem["metadatas"] else []
        sem_dists = raw_sem["distances"][0] if raw_sem["distances"] else []

        for cid, text, meta, dist in zip(sem_ids, sem_docs, sem_metas, sem_dists):
            sim = round(max(0.0, min(1.0, 1.0 - dist)), 3)
            semantic_candidates.append(
                {
                    "id": cid,
                    "text": text,
                    "metadata": meta,
                    "similarity": sim,
                }
            )

    bm25_index = get_bm25_index(
        collection_name=collection_name,
        persist_directory=persist_directory,
        client=active_client,
    )
    raw_lexical = bm25_index.query(query_text, top_k=candidate_depth)

    cached_embeddings = get_cached_embeddings(collection, collection_name)
    lexical_candidates: List[Dict[str, Any]] = []
    sem_lookup = {c["id"]: c["similarity"] for c in semantic_candidates}

    for lex_item in raw_lexical:
        cid = lex_item["id"]
        if cid in sem_lookup:
            sim = sem_lookup[cid]
        elif cid in cached_embeddings:
            doc_vec = cached_embeddings[cid]
            dot_sim = float(np.dot(query_embedding, doc_vec))
            sim = round(max(0.0, min(1.0, dot_sim)), 3)
        else:
            sim = 0.0

        item_copy = lex_item.copy()
        item_copy["similarity"] = sim
        lexical_candidates.append(item_copy)

    fused_pool = reciprocal_rank_fusion(
        semantic_candidates=semantic_candidates,
        lexical_candidates=lexical_candidates,
        k_rrf=k_rrf,
        w_sem=w_sem,
        w_lex=w_lex,
        top_k=candidate_depth,
    )

    if use_reranker:
        from rag.reranker import safe_rerank_candidates

        selected_candidates = safe_rerank_candidates(
            query_text=query_text,
            candidates=fused_pool,
            top_k=top_k,
        )
        active_mode = "rerank"
    else:
        selected_candidates = fused_pool[:top_k]
        active_mode = "hybrid"

    original_documents = [item["text"] for item in selected_candidates]
    original_candidates = [item.copy() for item in selected_candidates]

    if use_compressor:
        from rag.context_compressor import safe_compress_candidates

        comp_cands, _ = safe_compress_candidates(
            query_text=query_text,
            candidates=selected_candidates,
        )
        compressed_candidates = comp_cands
        compressed_documents = [item["text"] for item in comp_cands]
        active_mode = "compressed"
        documents = compressed_documents
        fused_to_return = comp_cands
    else:
        compressed_candidates = original_candidates
        compressed_documents = original_documents
        documents = original_documents
        fused_to_return = selected_candidates

    metadatas = [item["metadata"] for item in selected_candidates]
    similarities = [item.get("similarity", 0.0) for item in selected_candidates]
    top_similarity = similarities[0] if similarities else 0.0

    return {
        "query": query_text,
        "collection": collection_name,
        "documents": documents,
        "original_documents": original_documents,
        "compressed_documents": compressed_documents,
        "metadatas": metadatas,
        "similarities": similarities,
        "top_similarity": top_similarity,
        "mode": active_mode,
        "fused_candidates": fused_to_return,
        "original_candidates": original_candidates,
        "compressed_candidates": compressed_candidates,
        "candidate_pool": fused_pool,
    }
