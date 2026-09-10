# NykaaAssist — Product Requirements Document (PRD)

> **Document Status**: Authoritative Submission-Facing Product Specification  
> **Source Linkage**: Derived from and expanding upon the foundational root specification [`PRD.md`](PRD.md).  
> **Track**: E-commerce & Retail (Nykaa)  
> **Target System**: AI Customer Support Agent for Policy Inquiries and Order Operations

---

## Table of Contents

- [1. Background](#1-background)
- [2. Problem Statement](#2-problem-statement)
- [3. Product Goals & Non-Goals](#3-product-goals--non-goals)
- [4. Target Users & Personas](#4-target-users--personas)
- [5. Core User Problems & User Stories](#5-core-user-problems--user-stories)
- [6. Functional Requirements](#6-functional-requirements)
  - [Part A: Original Capstone Requirements (Tasks 1–16)](#part-a-original-capstone-requirements-tasks-116)
  - [Part B: Additional Engineering Enhancements (Tasks 17–26)](#part-b-additional-engineering-enhancements-tasks-1726)
- [7. Non-Functional Requirements](#7-non-functional-requirements)
- [8. Security & Guardrail Requirements](#8-security--guardrail-requirements)
- [9. Human Escalation (HITL) Requirements](#9-human-escalation-hitl-requirements)
- [10. Human Feedback Requirements](#10-human-feedback-requirements)
- [11. Evaluation Requirements](#11-evaluation-requirements)
- [12. Capstone Brief Alignment](#12-capstone-brief-alignment)
- [13. Related Documentation](#13-related-documentation)

---

## 1. Background

Nykaa is one of India's premier omnichannel beauty, personal care, fashion, and lifestyle e-commerce platforms, processing millions of orders annually. A substantial proportion of customer service contacts consists of repetitive, high-frequency inquiries:
- When does my return window close?
- How long does a Cash on Delivery (COD) refund take to credit to my bank?
- Where is my package and why is tracking delayed?
- What are the warranty terms on styling appliances?
- How do I claim for a damaged or leaking cosmetic item?

Currently, routine inquiries often default to human support agents, creating avoidable queue congestion, prolonged wait times for customers with urgent fulfillment issues, and high operational expenditure. **NykaaAssist** introduces an autonomous, highly grounded AI agent designed to resolve routine policy inquiries and operational lookups while maintaining enterprise-level data privacy and safety boundaries.

---

## 2. Problem Statement

Deploying conventional, unconstrained Large Language Models (LLMs) in e-commerce customer support poses severe commercial and brand risks:

1. **Hallucination of Commercial Policies**: Generative models may invent non-existent return windows (e.g., promising a 30-day return on opened perfumes when policy strictly forbids it), creating legal and customer relations liabilities.
2. **Stateless Disconnect**: Customers frequently engage in multi-turn dialogues (e.g., asking *"Where is order NYK-00001?"* followed by *"Can I return it?"*). A stateless bot fails to resolve conversational context.
3. **Data Privacy Exposure**: Support dialogues routinely contain sensitive Customer Personally Identifiable Information (PII)—including phone numbers, card digits, and UPI handles—which must never leak into application logs or model contexts.
4. **Adversarial Exploitation**: Malicious actors may attempt prompt injections to bypass company rules, exfiltrate backend system prompts, or manipulate refund logic.
5. **Operational Tool Silos**: Support agents require structured, protocol-standardized tool integrations (such as live order status, shipment tracking, and RMA generation) rather than hardcoded database scripts.

---

## 3. Product Goals & Non-Goals

### Product Goals
1. **Absolute Policy Grounding**: Answer customer policy questions using **only** retrieved, verified evidence from an authoritative knowledge base. If evidence is absent or insufficient, safely refuse rather than guess.
2. **Deterministic Operational Lookup**: Retrieve live order status, courier tracking, return eligibility, and loyalty status from backend operational data, computing an explicit, justified **SLA Escalation Signal** ($S_{esc}$) for delayed shipments.
3. **Multi-Turn Contextual Memory**: Retain conversational state, resolved pronouns, and order identifiers across turns within a session.
4. **Strict PII Protection**: Detect and mask fixed-format PII (Indian phone numbers, credit card numbers, last-4 patterns) before storage and logging.
5. **Zero-Cost Evaluator Reproducibility**: Guarantee 100% test reproducibility under an offline `MOCK_LLM=1` configuration with zero external network access and zero paid API dependencies.

### Non-Goals
- Real payment gateway transactions or live credit card chargebacks (simulated via deterministic records).
- Masking arbitrary free-text customer names or street addresses (explicitly out of scope per capstone brief, as no reliable keyless regex pattern exists).
- Autonomous mutation of production policies (customer feedback is collected strictly for human review and never modifies live policies automatically).
- Multi-language localization (English-only knowledge base for this capstone version).

---

## 4. Target Users & Personas

| Persona | Role | Core Need | Key Value Provided |
|---|---|---|---|
| **E-Commerce Customer** | End User | Immediate, accurate answers to policy and order inquiries without waiting in phone/chat queues | Instant grounded answers, clear return windows, real-time shipment ETAs |
| **Customer Support Agent** | Internal User | Triage high-risk cases without manually reviewing routine tickets | Automatic escalation score ($S_{esc}$) and structured `HumanSupportQueue` |
| **Capstone Evaluator / Grader** | Reviewer | Verify all architectural and functional claims deterministically without setup friction | Offline `MOCK_LLM=1` determinism, comprehensive test commands, and reproducible JSON metrics |

---

## 5. Core User Problems & User Stories

| ID | Persona | User Story | Acceptance Criteria |
|---|---|---|---|
| **U1** | Customer | As a customer, I want to ask *"What is the return window for footwear?"* | The agent responds with the exact 15-day policy, citing `return_window.md` with zero hallucinations. |
| **U2** | Customer | As a customer, I want to ask *"Where is order NYK-00001?"* | The agent returns live status (`Placed`), order value (₹2,301.65), and computes an SLA escalation score ($S_{esc} = 0.693$). |
| **U3** | Customer | As a customer, I want to ask a follow-up query *"Is it delayed?"* | The agent resolves pronoun `"it"` to `NYK-00001` from session memory without asking me to repeat the order number. |
| **U4** | Customer | As a customer, I want to accidentally paste my phone number (`9876543210`) | My phone number is masked to `***-***-3210` in session memory, logs, and model inputs. |
| **U5** | Support Agent | As a support agent, I want high-risk delayed shipments flagged automatically | Orders with $S_{esc} \ge 0.68$ are enqueued into `HumanSupportQueue` with high-priority reason and recommended action. |
| **U6** | Evaluator | As an evaluator, I want to interrupt a graph execution and resume it | The agent resumes from SQLite checkpoints without re-executing previously completed nodes. |
| **U7** | Evaluator | As an evaluator, I want to call operational tools via a standardized protocol | Operational tools are exposed via FastMCP and callable from an independent client process. |
| **U8** | Customer | As a customer, I want to ask a complex policy question with edge cases | The agent retrieves focused evidence, compresses irrelevant sentences, and validates claims before responding. |
| **U9** | Customer / Evaluator | As a user, I want to submit feedback (👍 / 👎) on an agent response | Ratings and scrubbed comments are stored in `feedback.sqlite` for human review without mutating production rules. |

---

## 6. Functional Requirements

### Part A: Original Capstone Requirements (Tasks 1–16)

These requirements represent the mandatory baseline criteria defined by the original capstone specification:

| Ref | Original Requirement | Capstone Brief Mapping | Implementation Module |
|---|---|---|---|
| **FR-01** | Generate $\ge$ 40 deterministic orders covering categories/statuses; delay rate in 10–30% | Part 1, Task 1 | `dataset.py` (50 orders seeded at 42; 24.0% delayed) |
| **FR-02** | Author $\ge$ 12 comprehensive knowledge base documents covering required policy topics | Part 1, Task 2 | `knowledge_base/*.md` (12 Markdown documents) |
| **FR-03** | Implement dual chunking strategies and index into two isolated ChromaDB collections | Part 1, Task 3 | `rag/chunking.py`, `rag/embed_index.py` |
| **FR-04** | Generate grounded answers from top-k retrieved chunks; calibrate fallback threshold | Part 1, Task 4 | `rag/generate.py` (0.35 similarity floor) |
| **FR-05** | Evaluate chunking strategies using Precision@3 and Recall@3; recommend production strategy | Part 1, Task 5 | `rag/evaluate_retrieval.py` (`nykaa_kb_fixed` recommended) |
| **FR-06** | Implement order lookup tool with designed SLA escalation score combining delay and recency | Part 2, Task 6 | `agent/tools.py` ($S_{esc} = 0.60 \cdot delay + 0.40 \cdot recency$) |
| **FR-07** | Implement LangGraph multi-node state machine with conditional intent routing | Part 2, Task 7 | `agent/graph.py` (Routing to policy vs order) |
| **FR-08** | Implement thread-persistent multi-turn conversation memory with clean-reset capability | Part 2, Task 8 | `agent/memory.py` (Thread isolation via `thread_id`) |
| **FR-09** | Validate every agent response against a structured Pydantic / JSON Schema | Part 2, Task 9 | `agent/schema.py` (`AgentResponse` model) |
| **FR-10** | Enforce input PII masking, prompt injection filtering, and output groundedness refusal | Part 2, Task 10 | `agent/guardrails.py` (Regex masking & injection filter) |
| **FR-11** | Deploy headless REST service with $\ge$ 2 endpoints and Pydantic validation | Part 3, Task 11 | `service/main.py` (`/health`, `/ask`, `/add-document`) |
| **FR-12** | Implement structured JSON-Lines logging with request trace IDs and pre-write PII masking | Part 3, Task 12 | `service/logging_utils.py` (`trace_id`, masked logs) |
| **FR-13** | Benchmark RAG Triad (Context Relevance, Groundedness, Answer Relevance) across 15 queries | Part 3, Task 13 | `eval/rag_triad.py` (`eval/rag_triad_results.json`) |
| **FR-14** | Expose operational tools via Model Context Protocol (FastMCP) with client invocation | Part 4, Task 14 | `mcp/server.py`, `mcp/client.py` |
| **FR-15** | Implement SQLite checkpointing with proven no-re-execution resume semantics | Part 4, Task 15 | `resilience/checkpoint_resume.py` (`checkpoints.sqlite`) |
| **FR-16** | Implement controlled query rewriter for pronoun resolution and intent clarification | Phase 5, Task 16 | `agent/rewrite.py` (Preserves raw query immutability) |

---

### Part B: Additional Engineering Enhancements (Tasks 17–26)

The following advanced capabilities were engineered as supplementary enhancements to achieve enterprise-grade reliability:

| Ref | Engineering Enhancement | Implementation Module | Verified Benchmark Outcome |
|---|---|---|---|
| **FR-17** | **Hybrid Retrieval & RRF**: Pure-Python Okapi BM25 + Dense ChromaDB with Cormack RRF ($k=60$) | `rag/hybrid.py`, `rag/lexical.py` | Top-1 retrieval accuracy improved from 80% to 100% (+25.0%) |
| **FR-18** | **Deterministic Cross Reranker**: 4-feature cross-scoring over top-10 candidate pool | `rag/reranker.py` | Overall RAG Triad score raised to 0.822 |
| **FR-19** | **Knowledge Gate Evidence Verification**: Multi-signal gate ($S_{sem}, S_{cov}, S_{rerank}$) with 1-step recovery | `rag/knowledge_gate.py` | 80% gain in adversarial fallback recall; 0% false fallbacks |
| **FR-20** | **Human-in-the-Loop (HITL) Node**: Automated triage routing high-risk queries to `HumanSupportQueue` | `agent/escalation.py` | 100% precision on delay triggers across 35-query benchmark |
| **FR-21** | **5-Tool FastMCP Ecosystem**: Order status, parcel tracking, return eligibility, RMA creation, loyalty | `mcp/server.py`, `agent/tools.py` | 100% routing accuracy, 0 duplicate RMAs under retries |
| **FR-22** | **Fault Resilience & Timeouts**: Exponential backoff, deterministic jitter, node/global timeouts | `resilience/retry_timeout.py` | 100% benchmark pass rate; clean timeout aborts |
| **FR-23** | **Master System Regression**: End-to-end integration and 50-query master regression benchmark | `eval/final_regression.py` | 50/50 queries pass (100.0%); 0 duplicate RMAs; 0 PII leaks |
| **FR-24** | **Sentence Context Compression**: Sentence reduction enforcing Non-Invention Invariant | `rag/context_compressor.py`| 11.1% context reduction; 100% verbatim source sentences |
| **FR-25** | **Answer Verification Agent**: Post-generation 3-way taxonomy (`PASS`/`REVISE`/`REJECT`) & claim repair | `agent/answer_verifier.py` | 0.0000 False PASS rate; 100% contradiction detection |
| **FR-26** | **Human Feedback Loop**: PII-scrubbed rating collection and review queue persistence | `agent/feedback.py` | Stored in `feedback.sqlite`; disagreement detection rate = 100% |
| **FR-UI** | **Streamlit Presentation Portal**: Conversational web UI with source chips and telemetry audit | `streamlit_app.py` | 10/10 customer journey scenarios execute cleanly |

---

## 7. Non-Functional Requirements

1. **Deterministic Reproducibility**: Execution under `MOCK_LLM=1` produces zero stochastic variance across all test runs.
2. **Execution Latency**:
   - Conversational greeting response latency: $< 50\text{ ms}$.
   - RAG retrieval, reranking, and grounded generation latency: $< 500\text{ ms}$.
   - Operational MCP tool execution latency: $< 200\text{ ms}$.
3. **Data Isolation & Concurrency**:
   - Checkpoint state transitions (`checkpoints.sqlite`) are thread-safe and isolated by `thread_id`.
   - Customer feedback records (`feedback.sqlite`) use dedicated WAL mode to prevent database locking under concurrent writes.
4. **Portability**: Operates without external database servers, cloud message brokers, or external container engines.

---

## 8. Security & Guardrail Requirements

1. **Input Guardrails**:
   - Regex masking of phone numbers, credit card numbers, last-4 card digits, UPI handles, and CVVs before passing input to agent nodes.
   - Regex filtering for prompt injection vectors (`"ignore previous instructions"`, `"act as system"`, `"dump prompt"`). Blocks request with zero confidence.
2. **Output Guardrails**:
   - Verification of semantic similarity floor ($0.35$). Fallback refusal emitted if evidence is ungrounded.
   - Final PII sanity sweep ensuring no unmasked phone numbers appear in final output strings.

---

## 9. Human Escalation (HITL) Requirements

The agent must automatically escalate to human support when:
- Order delay escalation score $S_{esc} \ge 0.68$.
- An order record is missing or not found in `orders.json`.
- A policy inquiry falls below the Knowledge Gate confidence threshold.
- The customer explicitly demands human intervention.

Escalations are placed into the in-memory, thread-safe `HumanSupportQueue` ($N=1000$) with priority, classification category, and sanitized conversation history.

---

## 10. Human Feedback Requirements

- Support binary customer satisfaction ratings: 👍 Helpful (`5`) and 👎 Not Helpful (`1`).
- Enforce mandatory PII scrubbing on customer comments before writing to `feedback.sqlite`.
- Associate feedback entries with request `trace_id` and verifier status.
- **Strict Non-Mutation Boundary**: Customer feedback records are stored for human review only and **never** autonomously modify production knowledge base files, prompt templates, or routing weights.

---

## 11. Evaluation Requirements

The system must satisfy empirical verification against established benchmark artifacts in `eval/`:
- Retrieval Precision@3 $\ge 0.55$, Recall@3 $= 1.00$.
- Hybrid Top-1 retrieval accuracy $= 1.00$ (+25% over dense-only baseline).
- Knowledge Gate adversarial fallback recall $\ge 0.85$.
- Master system regression pass rate $= 100.0\%$ across 50 queries.
- Duplicate RMAs generated under retry replay $= 0$.
- PII leak rate $= 0.0\%$.

---

## 12. Capstone Brief Alignment

This document directly aligns with the four evaluation phases of the original capstone brief:
- **Phase 1: Knowledge Base & RAG Foundations** $\to$ FR-01 through FR-05.
- **Phase 2: Agent Orchestration & Guardrails** $\to$ FR-06 through FR-10.
- **Phase 3: Production Service & Observability** $\to$ FR-11 through FR-13.
- **Phase 4: Resilience & Standardized Tool Protocols** $\to$ FR-14 through FR-16.

For technical architecture and database schemas, see [Technical Requirements Document (`TRD.md`)](./TRD.md). For design rationale, see [Research Paper (`RESEARCH_PAPER.md`)](./RESEARCH_PAPER.md).

---

## 13. Related Documentation

- [Central Documentation Hub (`README.md`)](./README.md)
- [Comprehensive Testing Manual (`TESTING.md`)](./TESTING.md)
- [Project Usage Guide (`USAGE.md`)](./USAGE.md)
- [Technical Requirements Document (`TRD.md`)](./TRD.md)
- [Technical Research & Engineering Rationale (`RESEARCH_PAPER.md`)](./RESEARCH_PAPER.md)
- [AI Architecture Specification (`AI_ARCHITECTURE.md`)](./AI_ARCHITECTURE.md)
- [Root Product Requirements (`PRD.md`)](PRD.md)
