import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
from rag.embed_index import (
    DEFAULT_PERSIST_DIR,
    FIXED_COLLECTION_NAME,
    get_chroma_client,
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "for",
    "with",
    "at",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "don",
    "now",
    "i",
    "me",
    "we",
    "us",
    "you",
    "it",
}

_BM25_CACHE: Dict[str, "BM25Index"] = {}


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = re.findall(r"\b[a-z0-9_]{2,}\b", cleaned)
    return [t for t in tokens if t not in STOPWORDS]


class BM25Index:
    def __init__(
        self,
        corpus_ids: List[str],
        corpus_texts: List[str],
        corpus_metadatas: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts
        self.corpus_metadatas = corpus_metadatas
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        self.doc_tokens = [tokenize(t) for t in corpus_texts]
        self.doc_lens = [len(dt) for dt in self.doc_tokens]
        self.avgdl = (
            sum(self.doc_lens) / float(self.corpus_size)
            if self.corpus_size > 0
            else 1.0
        )

        self.doc_freqs = Counter()
        for dt in self.doc_tokens:
            for term in set(dt):
                self.doc_freqs[term] += 1

        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1.0
            )

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query_text)
        if not q_tokens or self.corpus_size == 0:
            return []

        scores = []
        for idx in range(self.corpus_size):
            d_toks = self.doc_tokens[idx]
            d_len = self.doc_lens[idx]
            if d_len == 0:
                continue

            tf = Counter(d_toks)
            score = 0.0
            for qt in q_tokens:
                if qt in tf:
                    term_freq = tf[qt]
                    idf_val = self.idf.get(qt, 0.0)
                    denom = term_freq + self.k1 * (
                        1.0 - self.b + self.b * (d_len / self.avgdl)
                    )
                    score += idf_val * (term_freq * (self.k1 + 1.0)) / denom

            if score > 0.0:
                scores.append(
                    {
                        "id": self.corpus_ids[idx],
                        "text": self.corpus_texts[idx],
                        "metadata": self.corpus_metadatas[idx],
                        "score": round(score, 4),
                    }
                )

        scores.sort(key=lambda x: (-x["score"], x["id"]))
        return scores[:top_k]


def build_bm25_from_collection(collection: chromadb.Collection) -> BM25Index:
    raw_data = collection.get(include=["documents", "metadatas"])
    ids = raw_data.get("ids", [])
    documents = raw_data.get("documents", [])
    metadatas = raw_data.get("metadatas", [])
    return BM25Index(
        corpus_ids=ids,
        corpus_texts=documents,
        corpus_metadatas=metadatas,
    )


def get_bm25_index(
    collection_name: str = FIXED_COLLECTION_NAME,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    client: Optional[chromadb.PersistentClient] = None,
) -> BM25Index:
    cache_key = f"{persist_directory}::{collection_name}"
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]

    active_client = client or get_chroma_client(persist_directory)
    collection = active_client.get_collection(collection_name)
    index = build_bm25_from_collection(collection)
    _BM25_CACHE[cache_key] = index
    return index


def clear_bm25_cache() -> None:
    _BM25_CACHE.clear()
