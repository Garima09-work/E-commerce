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
    ANSWER_RELEVANCE_THRESHOLD,
    CONTEXT_RELEVANCE_THRESHOLD,
    DEFAULT_TEST_QUERIES_PATH,
    GROUNDEDNESS_THRESHOLD,
    OVERALL_TRIAD_THRESHOLD,
    calculate_answer_relevance,
    calculate_context_relevance,
    calculate_groundedness,
    load_test_queries,
)
from rag.embed_index import (
    DEFAULT_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    FIXED_COLLECTION_NAME,
    get_chroma_client,
)
from rag.generate import generate_grounded_answer, retrieve_context

DEFAULT_OUTPUT_JSON_PATH = str(ROOT_DIR / "eval" / "reranker_results.json")


def evaluate_single_query(
    query_item: Dict[str, Any],
    mode: str,
    collection_name: str,
    client: chromadb.PersistentClient,
    model: SentenceTransformer,
    retrieval_k: int = 3,
    generation_k: int = 2,
) -> Dict[str, Any]:
    query_id = query_item["id"]
    query_text = query_item["query"]
    expected_sources = query_item.get("expected_sources", [])
    topic = query_item.get("topic", "")

    retrieval_res = retrieve_context(
        query_text=query_text,
        collection_name=collection_name,
        top_k=retrieval_k,
        client=client,
        model=model,
        mode=mode,
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
        query_text=query_text,
        collection_name=collection_name,
        top_k=generation_k,
        client=client,
        model=model,
        mode=mode,
    )
    answer_res = generate_grounded_answer(
        query_text=query_text,
        collection_name=collection_name,
        top_k=generation_k,
        client=client,
        model=model,
        mode=mode,
    )

    retrieved_sources_gen = [
        m.get("source", "") for m in gen_retrieval.get("metadatas", [])
    ]
    similarities_gen = gen_retrieval.get("similarities", [])
    answer_text = answer_res.get("answer", "")
    confidence = answer_res.get("confidence", 0.0)
    response_type = answer_res.get("response_type", "policy_answer")

    cr_score, cr_audit = calculate_context_relevance(
        retrieved_sources=retrieved_sources_gen,
        expected_sources=expected_sources,
        similarities=similarities_gen,
    )
    g_score, g_audit = calculate_groundedness(
        answer=answer_text,
        retrieved_documents=gen_retrieval.get("documents", []),
        response_type=response_type,
    )
    ar_score, ar_audit = calculate_answer_relevance(
        query=query_text,
        answer=answer_text,
        response_type=response_type,
        model=model,
        is_out_of_scope=False,
    )
    triad_score = round((cr_score + g_score + ar_score) / 3.0, 3)

    return {
        "id": query_id,
        "query": query_text,
        "topic": topic,
        "expected_sources": expected_sources,
        "retrieved_sources_k3": retrieved_sources_k3,
        "retrieved_sources_gen": retrieved_sources_gen,
        "precision_at_3": precision_at_3,
        "recall_at_3": recall_at_3,
        "hit_top_1": hit_top_1,
        "confidence": confidence,
        "response_type": response_type,
        "context_relevance": cr_score,
        "groundedness": g_score,
        "answer_relevance": ar_score,
        "triad_score": triad_score,
        "cr_audit": cr_audit,
        "g_audit": g_audit,
        "ar_audit": ar_audit,
        "cr_pass": cr_score >= CONTEXT_RELEVANCE_THRESHOLD,
        "g_pass": g_score >= GROUNDEDNESS_THRESHOLD,
        "ar_pass": ar_score >= ANSWER_RELEVANCE_THRESHOLD,
        "overall_pass": (
            cr_score >= CONTEXT_RELEVANCE_THRESHOLD
            and g_score >= GROUNDEDNESS_THRESHOLD
            and ar_score >= ANSWER_RELEVANCE_THRESHOLD
            and triad_score >= OVERALL_TRIAD_THRESHOLD
        ),
        "answer_preview": answer_text[:120] if answer_text else "",
    }


def aggregate_stage_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    mean_p3 = round(sum(r["precision_at_3"] for r in results) / float(n), 3)
    mean_r3 = round(sum(r["recall_at_3"] for r in results) / float(n), 3)
    top1_acc = round(sum(1 for r in results if r["hit_top_1"]) / float(n), 3)

    mean_cr = round(sum(r["context_relevance"] for r in results) / float(n), 3)
    mean_g = round(sum(r["groundedness"] for r in results) / float(n), 3)
    mean_ar = round(sum(r["answer_relevance"] for r in results) / float(n), 3)
    mean_triad = round(sum(r["triad_score"] for r in results) / float(n), 3)

    cr_passes = sum(1 for r in results if r["cr_pass"])
    g_passes = sum(1 for r in results if r["g_pass"])
    ar_passes = sum(1 for r in results if r["ar_pass"])
    overall_passes = sum(1 for r in results if r["overall_pass"])

    return {
        "total_queries": n,
        "mean_precision_at_3": mean_p3,
        "mean_recall_at_3": mean_r3,
        "top_1_accuracy": top1_acc,
        "avg_context_relevance": mean_cr,
        "avg_groundedness": mean_g,
        "avg_answer_relevance": mean_ar,
        "avg_overall_triad": mean_triad,
        "cr_pass_rate": round(cr_passes / float(n), 3),
        "g_pass_rate": round(g_passes / float(n), 3),
        "ar_pass_rate": round(ar_passes / float(n), 3),
        "overall_pass_rate": round(overall_passes / float(n), 3),
    }


def compute_metrics_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        ("precision_at_3", "mean_precision_at_3"),
        ("recall_at_3", "mean_recall_at_3"),
        ("top_1_accuracy", "top_1_accuracy"),
        ("context_relevance", "avg_context_relevance"),
        ("groundedness", "avg_groundedness"),
        ("answer_relevance", "avg_answer_relevance"),
        ("overall_triad", "avg_overall_triad"),
    ]
    deltas = {}
    for label, metric_key in keys:
        b_val = before.get(metric_key, 0.0)
        a_val = after.get(metric_key, 0.0)
        d_val = round(a_val - b_val, 3)
        pct = round((d_val / b_val) * 100.0, 1) if b_val > 0 else 0.0
        deltas[label] = {
            "before": b_val,
            "after": a_val,
            "delta": d_val,
            "delta_percent": pct,
        }
    return deltas


def run_three_stage_evaluation(
    queries_path: str = DEFAULT_TEST_QUERIES_PATH,
    collection_name: str = FIXED_COLLECTION_NAME,
    output_json_path: str = DEFAULT_OUTPUT_JSON_PATH,
) -> Dict[str, Any]:
    queries = load_test_queries(queries_path)
    client = get_chroma_client(DEFAULT_PERSIST_DIR)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    semantic_results = []
    hybrid_results = []
    rerank_results = []

    for q in queries:
        sem_res = evaluate_single_query(
            query_item=q,
            mode="semantic",
            collection_name=collection_name,
            client=client,
            model=model,
        )
        semantic_results.append(sem_res)

        hyb_res = evaluate_single_query(
            query_item=q,
            mode="hybrid",
            collection_name=collection_name,
            client=client,
            model=model,
        )
        hybrid_results.append(hyb_res)

        rerank_res = evaluate_single_query(
            query_item=q,
            mode="rerank",
            collection_name=collection_name,
            client=client,
            model=model,
        )
        rerank_results.append(rerank_res)

    stage_a_metrics = aggregate_stage_metrics(semantic_results)
    stage_b_metrics = aggregate_stage_metrics(hybrid_results)
    stage_c_metrics = aggregate_stage_metrics(rerank_results)

    delta_sem_to_hyb = compute_metrics_delta(stage_a_metrics, stage_b_metrics)
    delta_hyb_to_rerank = compute_metrics_delta(stage_b_metrics, stage_c_metrics)
    delta_sem_to_rerank = compute_metrics_delta(stage_a_metrics, stage_c_metrics)

    per_query_comparison = []
    for s_res, h_res, r_res in zip(semantic_results, hybrid_results, rerank_results):
        qid = s_res["id"]
        per_query_comparison.append(
            {
                "id": qid,
                "topic": s_res["topic"],
                "expected_source": s_res["expected_sources"][0] if s_res["expected_sources"] else "",
                "semantic_retrieved_k3": s_res["retrieved_sources_k3"],
                "hybrid_retrieved_k3": h_res["retrieved_sources_k3"],
                "rerank_retrieved_k3": r_res["retrieved_sources_k3"],
                "semantic_top1": s_res["hit_top_1"],
                "hybrid_top1": h_res["hit_top_1"],
                "rerank_top1": r_res["hit_top_1"],
                "semantic_p3": s_res["precision_at_3"],
                "hybrid_p3": h_res["precision_at_3"],
                "rerank_p3": r_res["precision_at_3"],
                "semantic_cr": s_res["context_relevance"],
                "hybrid_cr": h_res["context_relevance"],
                "rerank_cr": r_res["context_relevance"],
                "semantic_triad": s_res["triad_score"],
                "hybrid_triad": h_res["triad_score"],
                "rerank_triad": r_res["triad_score"],
            }
        )

    full_payload = {
        "stage_a_semantic_baseline": stage_a_metrics,
        "stage_b_hybrid_candidate": stage_b_metrics,
        "stage_c_rerank_candidate": stage_c_metrics,
        "deltas": {
            "semantic_to_hybrid": delta_sem_to_hyb,
            "hybrid_to_rerank": delta_hyb_to_rerank,
            "semantic_to_rerank": delta_sem_to_rerank,
        },
        "per_query_comparison": per_query_comparison,
        "semantic_detailed_results": semantic_results,
        "hybrid_detailed_results": hybrid_results,
        "rerank_detailed_results": rerank_results,
    }

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("================================================================================")
    print("      NYKAA ASSIST TASK 18 — THREE-STAGE RETRIEVAL EVALUATION REPORT            ")
    print("================================================================================")
    print(f"{'Metric':<24} | {'Semantic (A)':<14} | {'Hybrid (B)':<14} | {'Rerank (C)':<14} | {'H->R Delta':<12}")
    print("-" * 88)
    metrics_to_print = [
        ("Precision@3", "mean_precision_at_3", "precision_at_3"),
        ("Recall@3", "mean_recall_at_3", "recall_at_3"),
        ("Top-1 Accuracy", "top_1_accuracy", "top_1_accuracy"),
        ("Context Relevance", "avg_context_relevance", "context_relevance"),
        ("Groundedness", "avg_groundedness", "groundedness"),
        ("Answer Relevance", "avg_answer_relevance", "answer_relevance"),
        ("Overall Triad", "avg_overall_triad", "overall_triad"),
    ]
    for label, m_key, d_key in metrics_to_print:
        val_a = stage_a_metrics[m_key]
        val_b = stage_b_metrics[m_key]
        val_c = stage_c_metrics[m_key]
        delta_info = delta_hyb_to_rerank[d_key]
        d_val = delta_info["delta"]
        sign = "+" if d_val > 0 else ""
        print(f"{label:<24} | {val_a:<14.3f} | {val_b:<14.3f} | {val_c:<14.3f} | {sign}{d_val:.3f} ({delta_info['delta_percent']}%)")
    print("================================================================================")

    return full_payload


def main() -> None:
    run_three_stage_evaluation()


if __name__ == "__main__":
    main()
