import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
from sentence_transformers import SentenceTransformer

from eval.rag_triad import (
    calculate_answer_relevance,
    calculate_context_relevance,
    calculate_groundedness,
    load_test_queries,
)
from rag.context_compressor import get_last_compression_result
from rag.embed_index import (
    DEFAULT_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    FIXED_COLLECTION_NAME,
    get_chroma_client,
)
from rag.generate import (
    FALLBACK_RESPONSE,
    generate_grounded_answer,
    retrieve_context,
)

DEFAULT_OUTPUT_JSON_PATH = str(ROOT_DIR / "eval" / "context_compression_results.json")
DEFAULT_TRANSCRIPT_PATH = str(ROOT_DIR / "transcripts" / "context_compression.txt")

OUT_OF_SCOPE_QUERIES = [
    {"id": "OOS_1", "query": "What is the capital of France?"},
    {"id": "OOS_2", "query": "What is the weather in Delhi today?"},
    {"id": "OOS_3", "query": "How do I write a quicksort algorithm in Python?"},
    {"id": "OOS_4", "query": "Who directed the movie Inception?"},
    {"id": "OOS_5", "query": "What is the stock price of Apple today?"},
]

ADVERSARIAL_QUERIES = [
    {"id": "ADV_1", "query": "Can I return a used foundation if the shade does not match my skin tone in store?"},
    {"id": "ADV_2", "query": "How do I return products bought from a physical Nykaa kiosk on the website?"},
    {"id": "ADV_3", "query": "Can I get a cash refund at my doorstep when the delivery partner picks up my parcel?"},
    {"id": "ADV_4", "query": "What is the warranty policy on counterfeit beauty products sold by third parties?"},
    {"id": "ADV_5", "query": "Will Nykaa pay for dermatologist consultation fees if an eye shadow causes irritation?"},
]


def evaluate_single_query_both_stages(
    query_item: Dict[str, Any],
    collection_name: str,
    client: chromadb.PersistentClient,
    model: SentenceTransformer,
    retrieval_k: int = 3,
    generation_k: int = 2,
) -> Dict[str, Any]:
    qid = query_item["id"]
    qtext = query_item["query"]
    expected_sources = query_item.get("expected_sources", [])
    topic = query_item.get("topic", "")

    retrieval_res = retrieve_context(
        query_text=qtext,
        collection_name=collection_name,
        top_k=retrieval_k,
        client=client,
        model=model,
        mode="rerank",
    )
    retrieved_sources_k3 = [
        m.get("source", "") for m in retrieval_res.get("metadatas", [])
    ]
    rel_cnt_k3 = sum(
        1 for src in retrieved_sources_k3 if src in expected_sources
    )
    precision_at_3 = round(rel_cnt_k3 / float(retrieval_k), 3)
    found_sources = set(retrieved_sources_k3) & set(expected_sources)
    recall_at_3 = (
        round(len(found_sources) / float(len(expected_sources)), 3)
        if expected_sources
        else 0.0
    )
    hit_top_1 = (
        len(retrieved_sources_k3) > 0
        and (retrieved_sources_k3[0] in expected_sources)
    )

    gen_retrieval = retrieve_context(
        query_text=qtext,
        collection_name=collection_name,
        top_k=generation_k,
        client=client,
        model=model,
        mode="rerank",
    )
    retrieved_sources_gen = [
        m.get("source", "") for m in gen_retrieval.get("metadatas", [])
    ]
    similarities_gen = gen_retrieval.get("similarities", [])

    answer_a = generate_grounded_answer(
        query_text=qtext,
        collection_name=collection_name,
        top_k=generation_k,
        client=client,
        model=model,
        mode="rerank",
        use_knowledge_gate=True,
        use_compressor=False,
    )
    ans_text_a = answer_a.get("answer", "")
    resp_type_a = answer_a.get("response_type", "policy_answer")

    cr_a, _ = calculate_context_relevance(
        retrieved_sources=retrieved_sources_gen,
        expected_sources=expected_sources,
        similarities=similarities_gen,
    )
    g_a, _ = calculate_groundedness(
        answer=ans_text_a,
        retrieved_documents=gen_retrieval.get("documents", []),
        response_type=resp_type_a,
    )
    ar_a, _ = calculate_answer_relevance(
        query=qtext,
        answer=ans_text_a,
        response_type=resp_type_a,
        model=model,
        is_out_of_scope=False,
    )
    triad_a = round((cr_a + g_a + ar_a) / 3.0, 3)

    answer_b = generate_grounded_answer(
        query_text=qtext,
        collection_name=collection_name,
        top_k=generation_k,
        client=client,
        model=model,
        mode="rerank",
        use_knowledge_gate=True,
        use_compressor=True,
    )
    ans_text_b = answer_b.get("answer", "")
    resp_type_b = answer_b.get("response_type", "policy_answer")
    comp_audit = get_last_compression_result()

    cr_b, _ = calculate_context_relevance(
        retrieved_sources=retrieved_sources_gen,
        expected_sources=expected_sources,
        similarities=similarities_gen,
    )
    g_b, _ = calculate_groundedness(
        answer=ans_text_b,
        retrieved_documents=gen_retrieval.get("documents", []),
        response_type=resp_type_b,
    )
    ar_b, _ = calculate_answer_relevance(
        query=qtext,
        answer=ans_text_b,
        response_type=resp_type_b,
        model=model,
        is_out_of_scope=False,
    )
    triad_b = round((cr_b + g_b + ar_b) / 3.0, 3)

    orig_chars = comp_audit.get("original_chars", len(ans_text_a))
    comp_chars = comp_audit.get("compressed_chars", len(ans_text_b))
    orig_words = comp_audit.get("original_words", len(ans_text_a.split()))
    comp_words = comp_audit.get("compressed_words", len(ans_text_b.split()))
    ratio = comp_audit.get("compression_ratio", 0.0)

    return {
        "id": qid,
        "query": qtext,
        "topic": topic,
        "expected_sources": expected_sources,
        "retrieved_sources_k3": retrieved_sources_k3,
        "precision_at_3": precision_at_3,
        "recall_at_3": recall_at_3,
        "hit_top_1": hit_top_1,
        "stage_a": {
            "response_type": resp_type_a,
            "answer": ans_text_a,
            "context_relevance": cr_a,
            "groundedness": g_a,
            "answer_relevance": ar_a,
            "triad_score": triad_a,
        },
        "stage_b": {
            "response_type": resp_type_b,
            "answer": ans_text_b,
            "context_relevance": cr_b,
            "groundedness": g_b,
            "answer_relevance": ar_b,
            "triad_score": triad_b,
            "compression_applied": comp_audit.get("compression_applied", False),
            "original_chars": orig_chars,
            "compressed_chars": comp_chars,
            "original_words": orig_words,
            "compressed_words": comp_words,
            "compression_ratio": ratio,
        },
    }


def run_full_task_24_benchmark() -> Dict[str, Any]:
    print("================================================================================")
    print("      NYKAA ASSIST TASK 24 — CONTEXT COMPRESSION EVALUATION REPORT              ")
    print("================================================================================")

    client = get_chroma_client(DEFAULT_PERSIST_DIR)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    queries = load_test_queries()

    in_scope_results = []
    print("\n--- 1. In-Scope Benchmark Evaluation (15 Queries) ---")
    for q in queries:
        item_res = evaluate_single_query_both_stages(
            query_item=q,
            collection_name=FIXED_COLLECTION_NAME,
            client=client,
            model=model,
        )
        in_scope_results.append(item_res)
        st_a = item_res["stage_a"]
        st_b = item_res["stage_b"]
        app_tag = "COMPRESSED" if st_b["compression_applied"] else "UNCHANGED"
        print(
            f"[{item_res['id']}] {app_tag} | Ratio: {st_b['compression_ratio'] * 100:.1f}% | "
            f"Triad A: {st_a['triad_score']:.3f} -> B: {st_b['triad_score']:.3f} | Query: '{item_res['query'][:55]}...'"
        )

    print("\n--- 2. Out-of-Scope Fallback Verification (5 Queries) ---")
    oos_correct_a = 0
    oos_correct_b = 0
    for q in OUT_OF_SCOPE_QUERIES:
        res_a = generate_grounded_answer(
            query_text=q["query"],
            collection_name=FIXED_COLLECTION_NAME,
            client=client,
            model=model,
            mode="rerank",
            use_knowledge_gate=True,
            use_compressor=False,
        )
        res_b = generate_grounded_answer(
            query_text=q["query"],
            collection_name=FIXED_COLLECTION_NAME,
            client=client,
            model=model,
            mode="rerank",
            use_knowledge_gate=True,
            use_compressor=True,
        )
        if res_a["response_type"] == "fallback":
            oos_correct_a += 1
        if res_b["response_type"] == "fallback":
            oos_correct_b += 1

    print(f"OOS Fallback Accuracy Stage A: {oos_correct_a}/5 | Stage B: {oos_correct_b}/5")

    print("\n--- 3. Adversarial / Edge Query Verification (5 Queries) ---")
    adv_fallback_a = 0
    adv_fallback_b = 0
    for q in ADVERSARIAL_QUERIES:
        res_a = generate_grounded_answer(
            query_text=q["query"],
            collection_name=FIXED_COLLECTION_NAME,
            client=client,
            model=model,
            mode="rerank",
            use_knowledge_gate=True,
            use_compressor=False,
        )
        res_b = generate_grounded_answer(
            query_text=q["query"],
            collection_name=FIXED_COLLECTION_NAME,
            client=client,
            model=model,
            mode="rerank",
            use_knowledge_gate=True,
            use_compressor=True,
        )
        if res_a["response_type"] == "fallback":
            adv_fallback_a += 1
        if res_b["response_type"] == "fallback":
            adv_fallback_b += 1

    print(f"Adversarial Fallback Stage A: {adv_fallback_a}/5 | Stage B: {adv_fallback_b}/5")

    n_queries = len(in_scope_results)
    p3_avg = round(sum(r["precision_at_3"] for r in in_scope_results) / n_queries, 3)
    r3_avg = round(sum(r["recall_at_3"] for r in in_scope_results) / n_queries, 3)
    top1_acc = round(sum(1 for r in in_scope_results if r["hit_top_1"]) / float(n_queries), 3)

    cr_a_avg = round(sum(r["stage_a"]["context_relevance"] for r in in_scope_results) / n_queries, 3)
    g_a_avg = round(sum(r["stage_a"]["groundedness"] for r in in_scope_results) / n_queries, 3)
    ar_a_avg = round(sum(r["stage_a"]["answer_relevance"] for r in in_scope_results) / n_queries, 3)
    triad_a_avg = round(sum(r["stage_a"]["triad_score"] for r in in_scope_results) / n_queries, 3)

    cr_b_avg = round(sum(r["stage_b"]["context_relevance"] for r in in_scope_results) / n_queries, 3)
    g_b_avg = round(sum(r["stage_b"]["groundedness"] for r in in_scope_results) / n_queries, 3)
    ar_b_avg = round(sum(r["stage_b"]["answer_relevance"] for r in in_scope_results) / n_queries, 3)
    triad_b_avg = round(sum(r["stage_b"]["triad_score"] for r in in_scope_results) / n_queries, 3)

    total_orig_chars = sum(r["stage_b"]["original_chars"] for r in in_scope_results)
    total_comp_chars = sum(r["stage_b"]["compressed_chars"] for r in in_scope_results)
    total_orig_words = sum(r["stage_b"]["original_words"] for r in in_scope_results)
    total_comp_words = sum(r["stage_b"]["compressed_words"] for r in in_scope_results)
    overall_char_ratio = (
        round(1.0 - (float(total_comp_chars) / float(total_orig_chars)), 4)
        if total_orig_chars > 0
        else 0.0
    )
    overall_word_ratio = (
        round(1.0 - (float(total_comp_words) / float(total_orig_words)), 4)
        if total_orig_words > 0
        else 0.0
    )

    fallback_recall_a = round((oos_correct_a + adv_fallback_a) / 10.0, 3)
    fallback_recall_b = round((oos_correct_b + adv_fallback_b) / 10.0, 3)
    false_fallback_a = round(
        sum(1 for r in in_scope_results if r["stage_a"]["response_type"] == "fallback") / float(n_queries),
        3,
    )
    false_fallback_b = round(
        sum(1 for r in in_scope_results if r["stage_b"]["response_type"] == "fallback") / float(n_queries),
        3,
    )

    print("\n================================================================================")
    print("                    SIDE-BY-SIDE EVALUATION METRICS                             ")
    print("================================================================================")
    print(f"Metric                     | Stage A (Baseline) | Stage B (+Compressor) | Delta")
    print(f"--------------------------------------------------------------------------------")
    print(f"Precision@3                | {p3_avg:.3f}              | {p3_avg:.3f}                 | {0.0:.3f}")
    print(f"Recall@3                   | {r3_avg:.3f}              | {r3_avg:.3f}                 | {0.0:.3f}")
    print(f"Top-1 Accuracy             | {top1_acc:.3f}              | {top1_acc:.3f}                 | {0.0:.3f}")
    print(f"Context Relevance          | {cr_a_avg:.3f}              | {cr_b_avg:.3f}                 | {cr_b_avg - cr_a_avg:+.3f}")
    print(f"Groundedness               | {g_a_avg:.3f}              | {g_b_avg:.3f}                 | {g_b_avg - g_a_avg:+.3f}")
    print(f"Answer Relevance           | {ar_a_avg:.3f}              | {ar_b_avg:.3f}                 | {ar_b_avg - ar_a_avg:+.3f}")
    print(f"Overall Triad              | {triad_a_avg:.3f}              | {triad_b_avg:.3f}                 | {triad_b_avg - triad_a_avg:+.3f}")
    print(f"Fallback Recall            | {fallback_recall_a:.3f}              | {fallback_recall_b:.3f}                 | {fallback_recall_b - fallback_recall_a:+.3f}")
    print(f"False Fallback Rate        | {false_fallback_a:.3f}              | {false_fallback_b:.3f}                 | {false_fallback_b - false_fallback_a:+.3f}")
    print(f"--------------------------------------------------------------------------------")
    print(f"Original Context Chars     | {total_orig_chars:<18} | {total_orig_chars:<21} | 0")
    print(f"Compressed Context Chars   | {total_orig_chars:<18} | {total_comp_chars:<21} | {total_comp_chars - total_orig_chars}")
    print(f"Char Compression Ratio     | 0.000              | {overall_char_ratio:.3f}                | {overall_char_ratio:+.3f} ({overall_char_ratio*100:.1f}%)")
    print(f"Original Context Words     | {total_orig_words:<18} | {total_orig_words:<21} | 0")
    print(f"Compressed Context Words   | {total_orig_words:<18} | {total_comp_words:<21} | {total_comp_words - total_orig_words}")
    print(f"Word Compression Ratio     | 0.000              | {overall_word_ratio:.3f}                | {overall_word_ratio:+.3f} ({overall_word_ratio*100:.1f}%)")
    print("================================================================================")

    output_data = {
        "metadata": {
            "benchmark_name": "Task 24 Context Compression Evaluation",
            "collection": FIXED_COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "total_in_scope_queries": n_queries,
            "total_out_of_scope_queries": len(OUT_OF_SCOPE_QUERIES),
            "total_adversarial_queries": len(ADVERSARIAL_QUERIES),
        },
        "summary_metrics": {
            "precision_at_3": p3_avg,
            "recall_at_3": r3_avg,
            "top_1_accuracy": top1_acc,
            "stage_a": {
                "context_relevance": cr_a_avg,
                "groundedness": g_a_avg,
                "answer_relevance": ar_a_avg,
                "overall_triad": triad_a_avg,
                "fallback_recall": fallback_recall_a,
                "false_fallback_rate": false_fallback_a,
            },
            "stage_b": {
                "context_relevance": cr_b_avg,
                "groundedness": g_b_avg,
                "answer_relevance": ar_b_avg,
                "overall_triad": triad_b_avg,
                "fallback_recall": fallback_recall_b,
                "false_fallback_rate": false_fallback_b,
            },
            "compression_efficiency": {
                "original_context_chars": total_orig_chars,
                "compressed_context_chars": total_comp_chars,
                "char_compression_ratio": overall_char_ratio,
                "char_reduction_percent": round(overall_char_ratio * 100.0, 2),
                "original_context_words": total_orig_words,
                "compressed_context_words": total_comp_words,
                "word_compression_ratio": overall_word_ratio,
                "word_reduction_percent": round(overall_word_ratio * 100.0, 2),
            },
        },
        "query_breakdown": in_scope_results,
    }

    with open(DEFAULT_OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nMachine-readable evaluation saved to: {DEFAULT_OUTPUT_JSON_PATH}")

    return output_data


if __name__ == "__main__":
    run_full_task_24_benchmark()
