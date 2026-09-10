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
from rag.generate import FALLBACK_RESPONSE, generate_grounded_answer, retrieve_context

DEFAULT_OUTPUT_JSON_PATH = str(ROOT_DIR / "eval" / "knowledge_gate_results.json")

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


def evaluate_single_in_scope_query(
    query_item: Dict[str, Any],
    use_gate: bool,
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
        query_text=query_text,
        collection_name=collection_name,
        top_k=generation_k,
        client=client,
        model=model,
        mode="rerank",
    )
    answer_res = generate_grounded_answer(
        query_text=query_text,
        collection_name=collection_name,
        top_k=generation_k,
        client=client,
        model=model,
        mode="rerank",
        use_knowledge_gate=use_gate,
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
        "is_fallback": response_type == "fallback",
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


def evaluate_negative_query(
    query_item: Dict[str, Any],
    use_gate: bool,
    collection_name: str,
    client: chromadb.PersistentClient,
    model: SentenceTransformer,
) -> Dict[str, Any]:
    query_id = query_item["id"]
    query_text = query_item["query"]

    answer_res = generate_grounded_answer(
        query_text=query_text,
        collection_name=collection_name,
        top_k=2,
        client=client,
        model=model,
        mode="rerank",
        use_knowledge_gate=use_gate,
    )

    response_type = answer_res.get("response_type", "policy_answer")
    is_fallback = response_type == "fallback" and answer_res.get("answer") == FALLBACK_RESPONSE
    is_unsupported_answer = response_type == "policy_answer"

    return {
        "id": query_id,
        "query": query_text,
        "response_type": response_type,
        "confidence": answer_res.get("confidence", 0.0),
        "is_fallback": is_fallback,
        "is_unsupported_answer": is_unsupported_answer,
        "sources": answer_res.get("sources", []),
    }


def aggregate_in_scope_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
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

    false_fallbacks = sum(1 for r in results if r["is_fallback"])
    false_fallback_rate = round(false_fallbacks / float(n), 3)

    return {
        "total_queries": n,
        "mean_precision_at_3": mean_p3,
        "mean_recall_at_3": mean_r3,
        "top_1_accuracy": top1_acc,
        "avg_context_relevance": mean_cr,
        "avg_groundedness": mean_g,
        "avg_answer_relevance": mean_ar,
        "avg_overall_triad": mean_triad,
        "false_fallback_rate": false_fallback_rate,
    }


def run_ab_gate_evaluation(
    queries_path: str = DEFAULT_TEST_QUERIES_PATH,
    collection_name: str = FIXED_COLLECTION_NAME,
    output_json_path: str = DEFAULT_OUTPUT_JSON_PATH,
) -> Dict[str, Any]:
    queries = load_test_queries(queries_path)
    client = get_chroma_client(DEFAULT_PERSIST_DIR)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    stage_a_in_scope = []
    stage_b_in_scope = []

    for q in queries:
        res_a = evaluate_single_in_scope_query(
            query_item=q,
            use_gate=False,
            collection_name=collection_name,
            client=client,
            model=model,
        )
        stage_a_in_scope.append(res_a)

        res_b = evaluate_single_in_scope_query(
            query_item=q,
            use_gate=True,
            collection_name=collection_name,
            client=client,
            model=model,
        )
        stage_b_in_scope.append(res_b)

    stage_a_oos = [
        evaluate_negative_query(q, False, collection_name, client, model)
        for q in OUT_OF_SCOPE_QUERIES
    ]
    stage_b_oos = [
        evaluate_negative_query(q, True, collection_name, client, model)
        for q in OUT_OF_SCOPE_QUERIES
    ]

    stage_a_adv = [
        evaluate_negative_query(q, False, collection_name, client, model)
        for q in ADVERSARIAL_QUERIES
    ]
    stage_b_adv = [
        evaluate_negative_query(q, True, collection_name, client, model)
        for q in ADVERSARIAL_QUERIES
    ]

    metrics_a = aggregate_in_scope_metrics(stage_a_in_scope)
    metrics_b = aggregate_in_scope_metrics(stage_b_in_scope)

    oos_fb_a = sum(1 for r in stage_a_oos if r["is_fallback"])
    oos_fb_b = sum(1 for r in stage_b_oos if r["is_fallback"])

    adv_fb_a = sum(1 for r in stage_a_adv if r["is_fallback"])
    adv_fb_b = sum(1 for r in stage_b_adv if r["is_fallback"])

    adv_unsupported_a = sum(1 for r in stage_a_adv if r["is_unsupported_answer"])
    adv_unsupported_b = sum(1 for r in stage_b_adv if r["is_unsupported_answer"])

    total_negatives = len(OUT_OF_SCOPE_QUERIES) + len(ADVERSARIAL_QUERIES)
    true_fb_a = oos_fb_a + adv_fb_a
    true_fb_b = oos_fb_b + adv_fb_b

    false_fb_a = sum(1 for r in stage_a_in_scope if r["is_fallback"])
    false_fb_b = sum(1 for r in stage_b_in_scope if r["is_fallback"])

    total_fb_a = true_fb_a + false_fb_a
    total_fb_b = true_fb_b + false_fb_b

    fb_precision_a = round(true_fb_a / float(total_fb_a), 3) if total_fb_a > 0 else 0.0
    fb_precision_b = round(true_fb_b / float(total_fb_b), 3) if total_fb_b > 0 else 0.0

    fb_recall_a = round(true_fb_a / float(total_negatives), 3)
    fb_recall_b = round(true_fb_b / float(total_negatives), 3)

    unsupported_rate_a = round(adv_unsupported_a / float(len(ADVERSARIAL_QUERIES)), 3)
    unsupported_rate_b = round(adv_unsupported_b / float(len(ADVERSARIAL_QUERIES)), 3)

    metrics_a["fallback_precision"] = fb_precision_a
    metrics_a["fallback_recall"] = fb_recall_a
    metrics_a["unsupported_answer_rate"] = unsupported_rate_a

    metrics_b["fallback_precision"] = fb_precision_b
    metrics_b["fallback_recall"] = fb_recall_b
    metrics_b["unsupported_answer_rate"] = unsupported_rate_b

    keys_to_compare = [
        "mean_precision_at_3",
        "mean_recall_at_3",
        "top_1_accuracy",
        "avg_context_relevance",
        "avg_groundedness",
        "avg_answer_relevance",
        "avg_overall_triad",
        "fallback_precision",
        "fallback_recall",
        "unsupported_answer_rate",
        "false_fallback_rate",
    ]
    deltas = {}
    for k in keys_to_compare:
        v_a = metrics_a[k]
        v_b = metrics_b[k]
        d = round(v_b - v_a, 3)
        pct = round((d / v_a) * 100.0, 1) if v_a > 0 else 0.0
        deltas[k] = {
            "stage_a": v_a,
            "stage_b": v_b,
            "delta": d,
            "delta_percent": pct,
        }

    full_payload = {
        "stage_a_baseline_reranker": metrics_a,
        "stage_b_knowledge_gate": metrics_b,
        "deltas": deltas,
        "stage_a_in_scope": stage_a_in_scope,
        "stage_b_in_scope": stage_b_in_scope,
        "stage_a_out_of_scope": stage_a_oos,
        "stage_b_out_of_scope": stage_b_oos,
        "stage_a_adversarial": stage_a_adv,
        "stage_b_adversarial": stage_b_adv,
    }

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)

    print("================================================================================")
    print("      NYKAA ASSIST TASK 19 — A/B KNOWLEDGE GATE EVALUATION REPORT               ")
    print("================================================================================")
    print(f"{'Metric':<26} | {'Stage A (Reranker)':<18} | {'Stage B (+Gate)':<18} | {'Delta':<14}")
    print("-" * 88)
    labels = [
        ("Precision@3", "mean_precision_at_3"),
        ("Recall@3", "mean_recall_at_3"),
        ("Top-1 Accuracy", "top_1_accuracy"),
        ("Context Relevance", "avg_context_relevance"),
        ("Groundedness", "avg_groundedness"),
        ("Answer Relevance", "avg_answer_relevance"),
        ("Overall Triad", "avg_overall_triad"),
        ("Fallback Precision", "fallback_precision"),
        ("Fallback Recall", "fallback_recall"),
        ("Unsupported Answer Rate", "unsupported_answer_rate"),
        ("False Fallback Rate", "false_fallback_rate"),
    ]
    for lbl, k in labels:
        val_a = metrics_a[k]
        val_b = metrics_b[k]
        d_info = deltas[k]
        sign = "+" if d_info["delta"] > 0 else ""
        print(f"{lbl:<26} | {val_a:<18.3f} | {val_b:<18.3f} | {sign}{d_info['delta']:.3f} ({d_info['delta_percent']}%)")
    print("================================================================================")

    return full_payload


def main() -> None:
    run_ab_gate_evaluation()


if __name__ == "__main__":
    main()
