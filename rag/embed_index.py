import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Any, Dict, List, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

from rag.chunking import split_fixed_size, split_by_sentence

DEFAULT_PERSIST_DIR = str(ROOT_DIR / "chroma_db")
FIXED_COLLECTION_NAME = "nykaa_kb_fixed"
SENTENCE_COLLECTION_NAME = "nykaa_kb_sentence"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
KNOWLEDGE_BASE_DIR = str(ROOT_DIR / "knowledge_base")


def load_knowledge_base_documents(
    kb_directory: str = KNOWLEDGE_BASE_DIR,
) -> List[Tuple[str, str]]:
    kb_path = Path(kb_directory)
    if not kb_path.exists() or not kb_path.is_dir():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_directory}")

    files = sorted(list(kb_path.glob("*.md")))
    if not files:
        raise ValueError(f"No markdown documents found in {kb_directory}")

    documents = []
    for file_path in files:
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            documents.append((file_path.name, content))

    return documents


def get_chroma_client(persist_directory: str = DEFAULT_PERSIST_DIR) -> chromadb.PersistentClient:
    storage_path = Path(persist_directory)
    storage_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(storage_path))


def index_chunks(
    client: chromadb.PersistentClient,
    model: SentenceTransformer,
    collection_name: str,
    chunks: List[Dict[str, Any]],
) -> chromadb.Collection:
    if not chunks:
        raise ValueError(f"No chunks provided for collection {collection_name}")

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


def build_indices(
    kb_directory: str = KNOWLEDGE_BASE_DIR,
    persist_directory: str = DEFAULT_PERSIST_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Dict[str, Any]:
    docs = load_knowledge_base_documents(kb_directory)

    fixed_chunks = []
    sentence_chunks = []

    for filename, content in docs:
        fixed_chunks.extend(split_fixed_size(content, filename))
        sentence_chunks.extend(split_by_sentence(content, filename))

    model = SentenceTransformer(model_name)
    client = get_chroma_client(persist_directory)

    fixed_collection = index_chunks(
        client=client,
        model=model,
        collection_name=FIXED_COLLECTION_NAME,
        chunks=fixed_chunks,
    )

    sentence_collection = index_chunks(
        client=client,
        model=model,
        collection_name=SENTENCE_COLLECTION_NAME,
        chunks=sentence_chunks,
    )

    return {
        "documents_loaded": len(docs),
        "fixed_chunks_count": len(fixed_chunks),
        "sentence_chunks_count": len(sentence_chunks),
        "fixed_collection_count": fixed_collection.count(),
        "sentence_collection_count": sentence_collection.count(),
    }


def query_collection(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query_text: str,
    n_results: int = 3,
) -> Dict[str, Any]:
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results


def run_smoke_tests(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> None:
    client = get_chroma_client(persist_directory)
    fixed_col = client.get_collection(FIXED_COLLECTION_NAME)
    sentence_col = client.get_collection(SENTENCE_COLLECTION_NAME)
    model = SentenceTransformer(model_name)

    sample_queries = [
        "Can I return this product?",
        "When will I get my refund?",
        "Can I cancel my order?",
        "My payment failed but money was deducted.",
        "I received a damaged product.",
        "Can I exchange my shoes for another size?",
        "How long does international delivery take?",
        "What is the warranty on my hair dryer?",
    ]

    print("\n========================================================")
    print("        CHROMA DUAL COLLECTION RETRIEVAL SMOKE TEST       ")
    print("========================================================")

    for query in sample_queries:
        fixed_res = query_collection(fixed_col, model, query, n_results=1)
        sent_res = query_collection(sentence_col, model, query, n_results=1)

        fixed_doc = fixed_res["metadatas"][0][0]["source"]
        fixed_dist = fixed_res["distances"][0][0]
        fixed_sim = 1.0 - fixed_dist

        sent_doc = sent_res["metadatas"][0][0]["source"]
        sent_dist = sent_res["distances"][0][0]
        sent_sim = 1.0 - sent_dist

        print(f"\nQuery: '{query}'")
        print(f"  [Fixed Collection]    Top: {fixed_doc:<26} (Cosine Sim: {fixed_sim:.3f})")
        print(f"  [Sentence Collection] Top: {sent_doc:<26} (Cosine Sim: {sent_sim:.3f})")

    print("\n========================================================")


def main() -> None:
    print("Starting dual chunking and ChromaDB indexing...")
    stats = build_indices()
    print("Indexing completed successfully.")
    print(f"  Documents Loaded         : {stats['documents_loaded']}")
    print(f"  Fixed Chunks Indexed     : {stats['fixed_chunks_count']} (Collection: {FIXED_COLLECTION_NAME})")
    print(f"  Sentence Chunks Indexed  : {stats['sentence_chunks_count']} (Collection: {SENTENCE_COLLECTION_NAME})")
    run_smoke_tests()


if __name__ == "__main__":
    main()
