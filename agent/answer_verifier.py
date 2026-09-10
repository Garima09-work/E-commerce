import os
import re
import sys
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.guardrails import detect_prompt_injection, mask_pii
from service.logging_utils import log_verification_event


class VerificationDecision(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    REJECT = "REJECT"


class VerificationMode(str, Enum):
    POLICY = "policy"
    OPERATIONAL = "operational"
    MIXED = "mixed"


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: VerificationDecision
    supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    supported_claims: List[str]
    unsupported_claims: List[str]
    contradicted_claims: List[str]
    missing_evidence: List[str]
    evidence_ids: List[str]
    reason: str
    verification_mode: VerificationMode
    verification_attempt: int = Field(default=1, ge=1)


NUMERICAL_TIME_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(business\s+days?|working\s+days?|days?|hours?|weeks?|months?|years?|inr|rs\.?|%|points?)\b",
    re.IGNORECASE,
)

NUMERICAL_VALUE_PATTERN = re.compile(
    r"\b(?:inr|rs\.?)\s*(\d+(?:\.\d+)?)\b|\b(\d+(?:\.\d+)?)\s*(?:inr|rupees?|days?|hours?|points?|%)\b",
    re.IGNORECASE,
)

ORDER_ID_PATTERN = re.compile(r"\bNYK-\d{5}\b", re.IGNORECASE)
CUSTOMER_ID_PATTERN = re.compile(r"\bCUST-\d{5}\b", re.IGNORECASE)
RMA_PATTERN = re.compile(r"\bRMA-[A-Z0-9-]+\b", re.IGNORECASE)

EXCLUSION_TERMS = [
    "non-returnable",
    "non returnable",
    "cannot be returned",
    "not returnable",
    "not eligible for return",
    "ineligible for return",
    "strictly non-returnable",
    "no return",
]

INCLUSION_TERMS = [
    "can be returned",
    "is returnable",
    "are returnable",
    "eligible for return",
    "qualify for a return",
    "accepted within",
    "allowed within",
]

RESTRICTED_CATEGORIES = [
    "intimate wear",
    "innerwear",
    "lingerie",
    "opened fragrance",
    "opened fragrances",
    "opened perfume",
    "opened perfumes",
    "customized personal care",
    "personal care item",
    "hygiene and safety",
]

FALLBACK_PHRASES = [
    "don't have enough grounded information",
    "not found in our records",
    "timed out",
    "could not be found",
    "cannot process this request as it violates our security policies",
]


def split_into_claims(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    clean = text.strip()
    if not clean:
        return []
    raw = re.split(r"(?<=[.!?])\s+", clean)
    claims = [c.strip() for c in raw if c.strip()]
    return claims if claims else [clean]


def tokenize_words(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text)]


def is_safe_fallback_response(answer: str) -> bool:
    if not answer:
        return True
    lower_ans = answer.lower()
    return any(phrase in lower_ans for phrase in FALLBACK_PHRASES)


def extract_numerical_tuples(text: str) -> List[Tuple[float, str]]:
    if not text:
        return []
    matches = NUMERICAL_TIME_PATTERN.findall(text)
    results = []
    for num_str, unit in matches:
        try:
            val = float(num_str)
            norm_unit = unit.lower().strip()
            norm_unit = re.sub(r"\s+", " ", norm_unit)
            if norm_unit.endswith("s") and not norm_unit.endswith("ss"):
                norm_unit = norm_unit.rstrip("s")
            results.append((val, norm_unit))
        except ValueError:
            continue
    return results


def check_operational_consistency(
    answer: str,
    tool_result: Optional[Dict[str, Any]],
    tool_name: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    if not tool_result or not isinstance(tool_result, dict):
        return [], [], []

    supported = []
    contradicted = []
    unsupported = []
    lower_ans = answer.lower()

    rec_id = tool_result.get("record_id")
    if rec_id:
        rec_matches = ORDER_ID_PATTERN.findall(answer)
        for found_id in rec_matches:
            if found_id.upper() == rec_id.upper():
                supported.append(f"Operational order ID {rec_id} verified")
            else:
                contradicted.append(f"Answer states order ID {found_id} but tool result is for {rec_id}")

    cid = tool_result.get("customer_id")
    if cid:
        cid_matches = CUSTOMER_ID_PATTERN.findall(answer)
        for found_cid in cid_matches:
            if found_cid.upper() == cid.upper():
                supported.append(f"Operational customer ID {cid} verified")
            else:
                contradicted.append(f"Answer states customer ID {found_cid} but tool result is for {cid}")

    status = tool_result.get("status")
    if status is not None:
        status_str = status.value if hasattr(status, "value") else str(status)
        status_lower = status_str.lower()
        if status_lower in lower_ans:
            supported.append(f"Operational status '{status_str}' matches tool output")
        else:
            conflicting_statuses = ["placed", "shipped", "delivered", "returned", "refunded", "cancelled"]
            conflicts = [st for st in conflicting_statuses if st in lower_ans and st != status_lower]
            if conflicts:
                contradicted.append(f"Answer claims status '{conflicts[0]}' but tool status is '{status_str}'")

    order_val = tool_result.get("order_value_inr")
    if order_val is not None and order_val > 0:
        val_matches = re.findall(r"\b(?:inr|rs\.?)\s*(\d+(?:\.\d+)?)\b", answer, re.IGNORECASE)
        val_matches += re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:inr|rupees?)\b", answer, re.IGNORECASE)
        for vm in val_matches:
            try:
                num_val = float(vm)
                if abs(num_val - float(order_val)) < 0.5:
                    supported.append(f"Order value INR {order_val:.2f} verified")
                else:
                    contradicted.append(f"Answer claims order value INR {num_val:.2f} but tool recorded INR {order_val:.2f}")
            except ValueError:
                pass

    if "eligible" in tool_result:
        is_eligible = bool(tool_result.get("eligible"))
        claims_ineligible = any(t in lower_ans for t in ["not eligible", "cannot be returned", "ineligible", "only delivered orders", "not found", "ineligible for return"])
        claims_eligible = (
            not claims_ineligible
            and any(t in lower_ans for t in ["is eligible", "eligible for return", "can be returned", "qualifies for return"])
        )
        if is_eligible and claims_eligible:
            supported.append("Return eligibility verified positive")
        elif not is_eligible and (claims_ineligible or not claims_eligible):
            supported.append("Return ineligibility verified negative")
        elif is_eligible and claims_ineligible:
            contradicted.append("Answer claims item is not eligible for return but tool confirmed eligibility")
        elif not is_eligible and claims_eligible:
            contradicted.append("Answer claims item is eligible for return but tool confirmed ineligibility")

    if "carrier" in tool_result:
        carrier = str(tool_result.get("carrier", "")).lower()
        if carrier and carrier != "n/a" and carrier in lower_ans:
            supported.append(f"Carrier '{carrier}' verified")

    if "rma_code" in tool_result:
        rma = str(tool_result.get("rma_code", "")).upper()
        if rma and rma != "N/A":
            found_rmas = RMA_PATTERN.findall(answer)
            for found_rma in found_rmas:
                if found_rma.upper() == rma:
                    supported.append(f"RMA code {rma} verified")
                else:
                    contradicted.append(f"Answer cites RMA {found_rma} but authoritative RMA is {rma}")

    if "tier" in tool_result:
        raw_tier = tool_result.get("tier", "")
        tier_str = raw_tier.value if hasattr(raw_tier, "value") else str(raw_tier)
        tier_val = tier_str.split(".")[-1].lower()
        if tier_val and tier_val in lower_ans:
            supported.append(f"Loyalty tier '{tier_val}' verified")
        else:
            other_tiers = [t for t in ["silver", "gold", "platinum"] if t != tier_val]
            found_other = [t for t in other_tiers if t in lower_ans]
            if found_other:
                contradicted.append(f"Answer states tier '{found_other[0]}' but customer is '{tier_val}' tier")

    if "points_balance" in tool_result:
        pts = tool_result.get("points_balance")
        if pts is not None:
            pts_matches = re.findall(r"\b(\d+)\s*points\b", answer, re.IGNORECASE)
            for pm in pts_matches:
                if int(pm) == int(pts):
                    supported.append(f"Points balance {pts} verified")
                else:
                    contradicted.append(f"Answer states {pm} points but customer has {pts} points")

    if "delayed_shipment" in tool_result:
        delayed = tool_result.get("delayed_shipment")
        if delayed is not None:
            if delayed and "delayed" in lower_ans and "not delayed" not in lower_ans:
                supported.append("Delayed shipment status verified")
            elif not delayed and ("not delayed" in lower_ans or "on time" in lower_ans):
                supported.append("Non-delayed shipment status verified")
            elif not delayed and "currently delayed" in lower_ans:
                contradicted.append("Answer claims order is delayed but tool indicates not delayed")
            elif delayed and "not delayed" in lower_ans:
                contradicted.append("Answer claims order is not delayed but tool indicates delayed")

    return supported, unsupported, contradicted


def check_exclusion_contradictions(
    claim: str,
    evidence_text: str,
) -> Optional[str]:
    clean_claim = claim.lower()
    clean_ev = evidence_text.lower()

    for cat in RESTRICTED_CATEGORIES:
        if cat in clean_claim:
            evidence_forbids = any(
                ex in clean_ev and cat in clean_ev
                for ex in EXCLUSION_TERMS
            )
            claim_permits = any(inc in clean_claim for inc in INCLUSION_TERMS)
            if evidence_forbids and claim_permits:
                return f"Claim permits return/refund for '{cat}', but evidence explicitly specifies non-returnable exclusion"
    return None


def verify_policy_claim(
    claim: str,
    evidence_items: List[Dict[str, Any]],
) -> Tuple[bool, bool, str]:
    if not claim.strip():
        return True, False, "Empty claim"

    is_inj, _ = detect_prompt_injection(claim)
    if is_inj:
        return False, True, "Prompt injection detected in claim"

    combined_evidence = " ".join(
        str(item.get("text", item.get("original_text", "")))
        for item in evidence_items
        if isinstance(item, dict)
    ).strip()

    if not combined_evidence:
        return False, False, "No evidence documents available to verify claim"

    contradiction_reason = check_exclusion_contradictions(claim, combined_evidence)
    if contradiction_reason:
        return False, True, contradiction_reason

    claim_numbers = extract_numerical_tuples(claim)
    evidence_numbers = extract_numerical_tuples(combined_evidence)

    if claim_numbers:
        for c_val, c_unit in claim_numbers:
            matching_num = any(
                abs(e_val - c_val) < 0.01 and (e_unit == c_unit or c_unit in e_unit or e_unit in c_unit)
                for e_val, e_unit in evidence_numbers
            )
            if not matching_num:
                related_ev_numbers = [
                    f"{e_val} {e_unit}"
                    for e_val, e_unit in evidence_numbers
                    if c_unit in e_unit or e_unit in c_unit
                ]
                if related_ev_numbers:
                    return False, False, f"Numerical mismatch: claim states {c_val} {c_unit}, but evidence specifies {', '.join(related_ev_numbers)}"
                return False, False, f"Unverified numerical entity: {c_val} {c_unit} not present in policy evidence"

    claim_tokens = tokenize_words(claim)
    ev_tokens = set(tokenize_words(combined_evidence))

    stopwords = {
        "the", "a", "an", "and", "or", "to", "for", "of", "with", "in", "on", "at",
        "by", "is", "are", "was", "were", "be", "been", "being", "this", "that",
        "these", "those", "you", "your", "our", "we", "can", "will", "may", "please",
        "should", "if", "not", "as", "from", "it", "its", "item", "items", "product", "products",
    }
    content_tokens = [t for t in claim_tokens if t not in stopwords and len(t) > 2]
    if not content_tokens:
        return True, False, "Claim contains only functional or conversational phrasing"

    overlap = sum(1 for t in content_tokens if t in ev_tokens)
    ratio = overlap / len(content_tokens)

    if ratio >= 0.50:
        return True, False, f"Claim grounded with {ratio:.1%} content token overlap"
    elif ratio >= 0.30 and len(content_tokens) <= 5:
        return True, False, f"Short claim grounded with {ratio:.1%} overlap"

    return False, False, f"Low groundedness overlap ({ratio:.1%}) against retrieved evidence"


def is_operational_claim(claim: str, tool_result: Optional[Dict[str, Any]] = None) -> bool:
    if not claim:
        return False
    if ORDER_ID_PATTERN.search(claim) or CUSTOMER_ID_PATTERN.search(claim) or RMA_PATTERN.search(claim):
        return True
    if tool_result and isinstance(tool_result, dict):
        rec_id = tool_result.get("record_id")
        if rec_id and str(rec_id).lower() in claim.lower():
            return True
        cid = tool_result.get("customer_id")
        if cid and str(cid).lower() in claim.lower():
            return True
        status = tool_result.get("status")
        if status:
            st_str = status.value if hasattr(status, "value") else str(status)
            if "order" in claim.lower() and st_str.lower() in claim.lower():
                return True
    return False


def has_repairable_numerical_mismatch(
    claim: str,
    evidence_items: List[Dict[str, Any]],
) -> bool:
    if not claim or not evidence_items:
        return False
    combined_ev = " ".join(
        str(item.get("text", item.get("original_text", "")))
        for item in evidence_items
        if isinstance(item, dict)
    )
    c_nums = extract_numerical_tuples(claim)
    e_nums = extract_numerical_tuples(combined_ev)
    if c_nums and e_nums:
        for _, c_unit in c_nums:
            for _, e_unit in e_nums:
                if c_unit in e_unit or e_unit in c_unit:
                    return True
    return False


def determine_verification_mode(
    route: Optional[str],
    tool_result: Optional[Dict[str, Any]],
    evidence_items: List[Dict[str, Any]],
) -> VerificationMode:
    has_tools = bool(tool_result and isinstance(tool_result, dict))
    has_policy = bool(evidence_items and len(evidence_items) > 0)

    if has_tools and has_policy:
        return VerificationMode.MIXED
    if has_tools or route == "order":
        return VerificationMode.OPERATIONAL
    return VerificationMode.POLICY


def repair_answer(
    draft_answer: str,
    evidence_items: List[Dict[str, Any]],
    tool_result: Optional[Dict[str, Any]],
    unsupported_claims: List[str],
    contradicted_claims: List[str],
) -> Optional[str]:
    if not draft_answer:
        return None

    claims = split_into_claims(draft_answer)
    combined_ev = " ".join(
        str(item.get("text", item.get("original_text", "")))
        for item in evidence_items
        if isinstance(item, dict)
    )

    ev_numbers = extract_numerical_tuples(combined_ev)
    ev_sentences = split_into_claims(combined_ev)

    repaired_claims = []

    for claim in claims:
        if any(claim.strip() == c.strip() for c in contradicted_claims):
            continue

        is_unsupported = any(claim.strip() == u.strip() for u in unsupported_claims)

        if is_unsupported:
            c_numbers = extract_numerical_tuples(claim)
            repaired_claim = claim
            repaired_num = False
            if c_numbers and ev_numbers:
                for c_val, c_unit in c_numbers:
                    for e_val, e_unit in ev_numbers:
                        if c_unit in e_unit or e_unit in c_unit:
                            pattern = re.compile(rf"\b{int(c_val) if c_val.is_integer() else c_val}\s*{re.escape(c_unit)}s?\b", re.IGNORECASE)
                            rep_val_str = f"{int(e_val) if e_val.is_integer() else e_val} {c_unit}"
                            if pattern.search(repaired_claim):
                                repaired_claim = pattern.sub(rep_val_str, repaired_claim)
                                repaired_num = True
            if repaired_num:
                repaired_claims.append(repaired_claim)
            else:
                continue
        else:
            repaired_claims.append(claim)

    if not repaired_claims and ev_sentences:
        candidate_sentence = ev_sentences[0].strip()
        if candidate_sentence:
            repaired_claims.append(candidate_sentence)

    if not repaired_claims:
        return None

    return " ".join(repaired_claims).strip()


def verify_answer(
    query: str,
    answer: str,
    evidence: Optional[Union[List[Dict[str, Any]], List[str]]] = None,
    tool_result: Optional[Dict[str, Any]] = None,
    tool_name: Optional[str] = None,
    route: Optional[str] = None,
    trace_id: Optional[str] = None,
    attempt: int = 1,
) -> VerificationResult:
    active_trace_id = trace_id or str(uuid.uuid4())
    norm_evidence: List[Dict[str, Any]] = []
    if evidence:
        for idx, item in enumerate(evidence):
            if isinstance(item, dict):
                norm_evidence.append(item)
            elif isinstance(item, str):
                norm_evidence.append({
                    "id": f"chunk_{idx}",
                    "text": item,
                    "metadata": {"source": "knowledge_base"},
                })

    mode = determine_verification_mode(route, tool_result, norm_evidence)
    evidence_ids = [
        str(item.get("id") or item.get("metadata", {}).get("source") or f"doc_{i}")
        for i, item in enumerate(norm_evidence)
    ]

    is_inj_query, _ = detect_prompt_injection(query or "")
    is_inj_ans, _ = detect_prompt_injection(answer or "")
    if is_inj_query or is_inj_ans:
        res = VerificationResult(
            decision=VerificationDecision.REJECT,
            supported=False,
            confidence=0.0,
            supported_claims=[],
            unsupported_claims=[],
            contradicted_claims=["Prompt injection directive detected in query or answer"],
            missing_evidence=[],
            evidence_ids=evidence_ids,
            reason="Prompt injection detected in verification payload",
            verification_mode=mode,
            verification_attempt=attempt,
        )
        log_verification_event(
            trace_id=active_trace_id,
            decision=res.decision.value,
            verification_mode=res.verification_mode.value,
            verification_attempt=attempt,
            supported_claims_count=0,
            unsupported_claims_count=0,
            contradicted_claims_count=1,
            evidence_ids=evidence_ids,
            repair_performed=False,
            reason=res.reason,
        )
        return res

    if is_safe_fallback_response(answer):
        res = VerificationResult(
            decision=VerificationDecision.PASS,
            supported=True,
            confidence=0.0,
            supported_claims=["Safe fallback response confirmed"],
            unsupported_claims=[],
            contradicted_claims=[],
            missing_evidence=[],
            evidence_ids=evidence_ids,
            reason="Safe fallback refusal verified as containing no ungrounded claims",
            verification_mode=mode,
            verification_attempt=attempt,
        )
        log_verification_event(
            trace_id=active_trace_id,
            decision=res.decision.value,
            verification_mode=res.verification_mode.value,
            verification_attempt=attempt,
            supported_claims_count=1,
            unsupported_claims_count=0,
            contradicted_claims_count=0,
            evidence_ids=evidence_ids,
            repair_performed=False,
            reason=res.reason,
        )
        return res

    supported_claims: List[str] = []
    unsupported_claims: List[str] = []
    contradicted_claims: List[str] = []
    missing_evidence: List[str] = []

    if mode in (VerificationMode.OPERATIONAL, VerificationMode.MIXED) and tool_result:
        op_supp, op_unsupp, op_contra = check_operational_consistency(
            answer=answer,
            tool_result=tool_result,
            tool_name=tool_name,
        )
        supported_claims.extend(op_supp)
        unsupported_claims.extend(op_unsupp)
        contradicted_claims.extend(op_contra)

    if mode in (VerificationMode.POLICY, VerificationMode.MIXED):
        if not norm_evidence:
            missing_evidence.append("Knowledge base policy evidence absent")
            unsupported_claims.append(answer)
        else:
            claims = split_into_claims(answer)
            for c in claims:
                if mode == VerificationMode.MIXED and is_operational_claim(c, tool_result):
                    continue
                is_supp, is_contra, c_reason = verify_policy_claim(c, norm_evidence)
                if is_contra:
                    contradicted_claims.append(c)
                elif is_supp:
                    supported_claims.append(c)
                else:
                    unsupported_claims.append(c)

    has_repairable = any(
        has_repairable_numerical_mismatch(c, norm_evidence)
        for c in unsupported_claims
    )

    if contradicted_claims:
        decision = VerificationDecision.REJECT
        supported = False
        reason = f"Answer materially contradicts authoritative evidence ({len(contradicted_claims)} contradiction(s))"
    elif unsupported_claims and supported_claims:
        if attempt < 2:
            decision = VerificationDecision.REVISE
            supported = False
            reason = f"Answer partially grounded but contains {len(unsupported_claims)} repairable unsupported claim(s)"
        else:
            decision = VerificationDecision.REJECT
            supported = False
            reason = f"Answer contains unresolved unsupported claims after {attempt} verification attempt(s)"
    elif not supported_claims and unsupported_claims:
        if has_repairable and attempt < 2:
            decision = VerificationDecision.REVISE
            supported = False
            reason = f"Numerical mismatch against evidence in {len(unsupported_claims)} claim(s), repairable from retrieved context"
        else:
            decision = VerificationDecision.REJECT
            supported = False
            reason = f"Answer completely unsupported by authoritative evidence ({len(unsupported_claims)} unverified claim(s))"
    elif supported_claims and not unsupported_claims and not contradicted_claims:
        decision = VerificationDecision.PASS
        supported = True
        reason = "All factual claims verified against authoritative evidence and tool results"
    else:
        decision = VerificationDecision.REJECT
        supported = False
        reason = "Verification could not establish evidence grounding for generated answer"

    confidence = round(1.0 if supported else 0.0, 3)

    result = VerificationResult(
        decision=decision,
        supported=supported,
        confidence=confidence,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
        contradicted_claims=contradicted_claims,
        missing_evidence=missing_evidence,
        evidence_ids=evidence_ids,
        reason=reason,
        verification_mode=mode,
        verification_attempt=attempt,
    )

    log_verification_event(
        trace_id=active_trace_id,
        decision=result.decision.value,
        verification_mode=result.verification_mode.value,
        verification_attempt=attempt,
        supported_claims_count=len(supported_claims),
        unsupported_claims_count=len(unsupported_claims),
        contradicted_claims_count=len(contradicted_claims),
        evidence_ids=evidence_ids,
        repair_performed=False,
        reason=result.reason,
    )

    return result


def safe_verify_and_repair(
    query: str,
    answer: str,
    evidence: Optional[Union[List[Dict[str, Any]], List[str]]] = None,
    tool_result: Optional[Dict[str, Any]] = None,
    tool_name: Optional[str] = None,
    route: Optional[str] = None,
    trace_id: Optional[str] = None,
    max_attempts: int = 2,
) -> Tuple[VerificationResult, str]:
    active_trace = trace_id or str(uuid.uuid4())
    current_answer = answer
    norm_evidence: List[Dict[str, Any]] = []
    if evidence:
        for idx, item in enumerate(evidence):
            if isinstance(item, dict):
                norm_evidence.append(item)
            elif isinstance(item, str):
                norm_evidence.append({
                    "id": f"chunk_{idx}",
                    "text": item,
                    "metadata": {"source": "knowledge_base"},
                })

    for attempt in range(1, max_attempts + 1):
        v_res = verify_answer(
            query=query,
            answer=current_answer,
            evidence=norm_evidence,
            tool_result=tool_result,
            tool_name=tool_name,
            route=route,
            trace_id=active_trace,
            attempt=attempt,
        )
        if v_res.decision == VerificationDecision.PASS:
            return v_res, current_answer
        if v_res.decision == VerificationDecision.REJECT:
            return v_res, current_answer

        if v_res.decision == VerificationDecision.REVISE and attempt < max_attempts:
            repaired = repair_answer(
                draft_answer=current_answer,
                evidence_items=norm_evidence,
                tool_result=tool_result,
                unsupported_claims=v_res.unsupported_claims,
                contradicted_claims=v_res.contradicted_claims,
            )
            if repaired and repaired.strip() and repaired.strip() != current_answer.strip():
                current_answer = repaired
                continue
            else:
                final_res = VerificationResult(
                    decision=VerificationDecision.REJECT,
                    supported=False,
                    confidence=0.0,
                    supported_claims=v_res.supported_claims,
                    unsupported_claims=v_res.unsupported_claims,
                    contradicted_claims=v_res.contradicted_claims,
                    missing_evidence=v_res.missing_evidence,
                    evidence_ids=v_res.evidence_ids,
                    reason="Deterministic repair could not produce grounded text from available evidence",
                    verification_mode=v_res.verification_mode,
                    verification_attempt=attempt + 1,
                )
                return final_res, current_answer

    final_res = VerificationResult(
        decision=VerificationDecision.REJECT,
        supported=False,
        confidence=0.0,
        supported_claims=[],
        unsupported_claims=[current_answer],
        contradicted_claims=[],
        missing_evidence=[],
        evidence_ids=[],
        reason=f"Exhausted maximum verification attempts ({max_attempts}) without establishing groundedness",
        verification_mode=VerificationMode.POLICY,
        verification_attempt=max_attempts,
    )
    return final_res, current_answer
