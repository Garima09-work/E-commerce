import re
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_FIXED_CHUNK_SIZE = 50
DEFAULT_FIXED_OVERLAP = 10
DEFAULT_SENTENCES_PER_CHUNK = 2


def extract_clean_text(markdown_content: str) -> str:
    lines = [
        line.strip()
        for line in markdown_content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return " ".join(lines)


def split_fixed_size(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
    overlap: int = DEFAULT_FIXED_OVERLAP,
) -> List[Dict[str, Any]]:
    clean_text = extract_clean_text(text) if "#" in text else text.strip()
    words = clean_text.split()
    if not words:
        return []

    stem = Path(source).stem
    if len(words) <= chunk_size:
        return [
            {
                "id": f"{stem}_fixed_000",
                "text": " ".join(words),
                "metadata": {
                    "source": source,
                    "strategy": "fixed",
                    "chunk_index": 0,
                },
            }
        ]

    step = max(1, chunk_size - overlap)
    chunks = []
    chunk_index = 0

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append(
            {
                "id": f"{stem}_fixed_{chunk_index:03d}",
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "strategy": "fixed",
                    "chunk_index": chunk_index,
                },
            }
        )
        chunk_index += 1
        if start + chunk_size >= len(words):
            break

    return chunks


def split_by_sentence(
    text: str,
    source: str,
    sentences_per_chunk: int = DEFAULT_SENTENCES_PER_CHUNK,
) -> List[Dict[str, Any]]:
    clean_text = extract_clean_text(text) if "#" in text else text.strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean_text)
        if sentence.strip()
    ]
    if not sentences:
        return []

    stem = Path(source).stem
    chunks = []
    chunk_index = 0

    for start in range(0, len(sentences), sentences_per_chunk):
        sentence_group = sentences[start : start + sentences_per_chunk]
        if not sentence_group:
            break
        chunk_text = " ".join(sentence_group)
        chunks.append(
            {
                "id": f"{stem}_sentence_{chunk_index:03d}",
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "strategy": "sentence",
                    "chunk_index": chunk_index,
                },
            }
        )
        chunk_index += 1

    return chunks
