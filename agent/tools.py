import json
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.schema import (
    LoyaltyStatusResult,
    LoyaltyTier,
    OrderStatusResult,
    ReturnEligibilityResult,
    ReturnRequestResult,
    ShipmentTrackingResult,
)

DEFAULT_ORDERS_PATH = str(ROOT_DIR / "orders.json")
WEIGHT_DELAY = 0.60
WEIGHT_RECENCY = 0.40
DEFAULT_MAX_DAYS = 30
ESCALATION_THRESHOLD_80TH_PERCENTILE = 0.68

ORDER_ID_REGEX = re.compile(r"^NYK-\d{5}$")
CUSTOMER_ID_REGEX = re.compile(r"^CUST-\d{5}$")

RETURN_REGISTRY_LOCK = threading.Lock()
ACTIVE_RETURN_REQUESTS: Dict[str, Dict[str, Any]] = {}


def clear_return_requests() -> None:
    with RETURN_REGISTRY_LOCK:
        ACTIVE_RETURN_REQUESTS.clear()


def load_orders_dataset(file_path: str = DEFAULT_ORDERS_PATH) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Orders dataset not found at: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        orders = json.load(f)

    return orders


def compute_escalation_score(
    delayed_shipment: bool,
    days_since_created: int,
    max_days: int = DEFAULT_MAX_DAYS,
) -> float:
    delayed_flag = 1.0 if delayed_shipment else 0.0
    safe_max_days = max(1, max_days)
    normalized_recency = min(1.0, max(0.0, float(days_since_created) / float(safe_max_days)))

    score = round((WEIGHT_DELAY * delayed_flag) + (WEIGHT_RECENCY * normalized_recency), 3)
    return max(0.0, min(1.0, score))


def check_order_status(
    record_id: str,
    file_path: str = DEFAULT_ORDERS_PATH,
) -> Dict[str, Any]:
    orders = load_orders_dataset(file_path)
    normalized_id = record_id.strip().upper()

    matching_order = None
    for order in orders:
        if order.get("record_id", "").upper() == normalized_id:
            matching_order = order
            break

    if matching_order is None:
        raw_res = {
            "record_id": record_id,
            "status": "Not Found",
            "order_value_inr": None,
            "escalation_score": None,
            "error": f"Order {record_id} not found.",
        }
        return raw_res

    escalation_score = compute_escalation_score(
        delayed_shipment=bool(matching_order.get("delayed_shipment", False)),
        days_since_created=int(matching_order.get("days_since_created", 0)),
        max_days=DEFAULT_MAX_DAYS,
    )

    validated = OrderStatusResult(
        record_id=matching_order["record_id"],
        status=matching_order["status"],
        order_value_inr=float(matching_order["order_value_inr"]),
        escalation_score=escalation_score,
        delayed_shipment=bool(matching_order.get("delayed_shipment", False)),
        error=None,
    )
    return validated.model_dump()


def track_shipment(
    record_id: str,
    file_path: str = DEFAULT_ORDERS_PATH,
) -> Dict[str, Any]:
    normalized_id = record_id.strip().upper()
    if not ORDER_ID_REGEX.match(normalized_id):
        return {
            "record_id": record_id,
            "status": "Invalid Identifier",
            "carrier": "N/A",
            "tracking_number": "N/A",
            "current_location": "N/A",
            "estimated_delivery": "N/A",
            "delayed_shipment": False,
            "error": f"Invalid order ID format: {record_id}",
        }

    orders = load_orders_dataset(file_path)
    matching_order = None
    for order in orders:
        if order.get("record_id", "").upper() == normalized_id:
            matching_order = order
            break

    if matching_order is None:
        return {
            "record_id": normalized_id,
            "status": "Not Found",
            "carrier": "N/A",
            "tracking_number": "N/A",
            "current_location": "N/A",
            "estimated_delivery": "N/A",
            "delayed_shipment": False,
            "error": f"Order {normalized_id} not found.",
        }

    status = matching_order.get("status", "Placed")
    delayed = bool(matching_order.get("delayed_shipment", False))

    if status == "Placed":
        carrier = "BlueDart Express"
        tracking_num = f"BD-{normalized_id}"
        location = "Warehouse Processing Facility"
        eta = "In 4 business days"
    elif status == "Shipped":
        carrier = "BlueDart Express"
        tracking_num = f"BD-{normalized_id}"
        if delayed:
            location = "Regional Transit Hub (Logistics Delay Notice)"
            eta = "Delayed - Revised ETA: In 3 business days"
        else:
            location = "In Transit - Out for Regional Delivery"
            eta = "In 1-2 business days"
    elif status == "Delivered":
        carrier = "BlueDart Express"
        tracking_num = f"BD-{normalized_id}"
        location = "Delivered to Customer Address"
        eta = "Delivered"
    elif status == "Returned":
        carrier = "BlueDart Reverse Logistics"
        tracking_num = f"BDR-{normalized_id}"
        location = "Returned to Central Warehouse"
        eta = "Return Completed"
    elif status == "Refunded":
        carrier = "N/A"
        tracking_num = "N/A"
        location = "N/A"
        eta = "Refund Processed"
    else:
        carrier = "BlueDart Express"
        tracking_num = f"BD-{normalized_id}"
        location = "Order Processed"
        eta = "Standard Delivery (3-5 business days)"

    validated = ShipmentTrackingResult(
        record_id=normalized_id,
        status=status,
        carrier=carrier,
        tracking_number=tracking_num,
        current_location=location,
        estimated_delivery=eta,
        delayed_shipment=delayed,
        error=None,
    )
    return validated.model_dump()


def check_return_status(
    record_id: str,
    file_path: str = DEFAULT_ORDERS_PATH,
) -> Dict[str, Any]:
    normalized_id = record_id.strip().upper()
    if not ORDER_ID_REGEX.match(normalized_id):
        return {
            "record_id": record_id,
            "eligible": False,
            "category": None,
            "days_since_delivery": None,
            "allowed_window_days": None,
            "reason": f"Invalid order ID format: {record_id}",
            "error": f"Invalid order ID format: {record_id}",
        }

    orders = load_orders_dataset(file_path)
    matching_order = None
    for order in orders:
        if order.get("record_id", "").upper() == normalized_id:
            matching_order = order
            break

    if matching_order is None:
        return {
            "record_id": normalized_id,
            "eligible": False,
            "category": None,
            "days_since_delivery": None,
            "allowed_window_days": None,
            "reason": f"Order {normalized_id} not found.",
            "error": f"Order {normalized_id} not found.",
        }

    category = matching_order.get("category", "")
    days = int(matching_order.get("days_since_created", 0))
    status = matching_order.get("status", "")

    if category in ("Beauty", "Apparel", "Footwear"):
        allowed_window = 15
    elif category in ("Electronics", "Home"):
        allowed_window = 7
    else:
        allowed_window = 15

    if status != "Delivered":
        eligible = False
        reason = f"Order status is '{status}'. Only delivered orders are eligible for return."
    elif days <= allowed_window:
        eligible = True
        reason = f"Order is eligible for return within the {allowed_window}-day window ({days} days elapsed)."
    else:
        eligible = False
        reason = f"Return window expired ({days} days elapsed, maximum allowed window is {allowed_window} days for {category})."

    validated = ReturnEligibilityResult(
        record_id=normalized_id,
        eligible=eligible,
        category=category,
        days_since_delivery=days if status == "Delivered" else None,
        allowed_window_days=allowed_window,
        reason=reason,
        error=None,
    )
    return validated.model_dump()


def create_return_request(
    record_id: str,
    reason: str,
    file_path: str = DEFAULT_ORDERS_PATH,
) -> Dict[str, Any]:
    normalized_id = record_id.strip().upper()
    if not ORDER_ID_REGEX.match(normalized_id):
        return {
            "record_id": record_id,
            "request_id": f"REQ-ERR-{record_id}",
            "status": "Failed",
            "rma_code": "N/A",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "error": f"Invalid order ID format: {record_id}",
        }

    clean_reason = reason.strip() if isinstance(reason, str) else ""
    if len(clean_reason) < 3 or len(clean_reason) > 200:
        return {
            "record_id": normalized_id,
            "request_id": f"REQ-ERR-{normalized_id}",
            "status": "Failed",
            "rma_code": "N/A",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": clean_reason,
            "error": "Return reason must be between 3 and 200 characters.",
        }

    with RETURN_REGISTRY_LOCK:
        if normalized_id in ACTIVE_RETURN_REQUESTS:
            existing = dict(ACTIVE_RETURN_REQUESTS[normalized_id])
            return existing

    eligibility = check_return_status(normalized_id, file_path)
    if not eligibility.get("eligible", False):
        return {
            "record_id": normalized_id,
            "request_id": f"REQ-REJ-{normalized_id}",
            "status": "Rejected",
            "rma_code": "N/A",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": clean_reason,
            "error": f"Order ineligible for return: {eligibility.get('reason', 'Policy check failed')}",
        }

    suffix_num = abs(hash(normalized_id + clean_reason)) % 10000
    rma_code = f"RMA-{normalized_id}-{suffix_num:04d}"
    request_id = f"RET-{normalized_id}-{abs(hash(normalized_id)) % 1000:03d}"
    created_at = datetime.now(timezone.utc).isoformat()

    new_record = {
        "record_id": normalized_id,
        "request_id": request_id,
        "status": "Initiated",
        "rma_code": rma_code,
        "created_at": created_at,
        "reason": clean_reason,
        "error": None,
    }

    with RETURN_REGISTRY_LOCK:
        ACTIVE_RETURN_REQUESTS[normalized_id] = new_record

    validated = ReturnRequestResult.model_validate(new_record)
    return validated.model_dump()


def loyalty_status(
    customer_id: str,
    file_path: str = DEFAULT_ORDERS_PATH,
) -> Dict[str, Any]:
    normalized_cid = customer_id.strip().upper()
    if not CUSTOMER_ID_REGEX.match(normalized_cid):
        return {
            "customer_id": customer_id,
            "tier": "Silver",
            "points_balance": 0,
            "lifetime_spend_inr": 0.0,
            "error": f"Invalid customer ID format: {customer_id}",
        }

    mapped_order_id = f"NYK-{normalized_cid[5:]}"
    orders = load_orders_dataset(file_path)
    matching_order = None
    for order in orders:
        if order.get("record_id", "").upper() == mapped_order_id:
            matching_order = order
            break

    if matching_order is None:
        validated = LoyaltyStatusResult(
            customer_id=normalized_cid,
            tier=LoyaltyTier.SILVER,
            points_balance=0,
            lifetime_spend_inr=0.0,
            error=None,
        )
        return validated.model_dump()

    spend = float(matching_order.get("order_value_inr", 0.0))
    points = int(spend // 100)

    if spend >= 5000.0:
        tier = LoyaltyTier.PLATINUM
    elif spend >= 1000.0:
        tier = LoyaltyTier.GOLD
    else:
        tier = LoyaltyTier.SILVER

    validated = LoyaltyStatusResult(
        customer_id=normalized_cid,
        tier=tier,
        points_balance=points,
        lifetime_spend_inr=spend,
        error=None,
    )
    return validated.model_dump()


def run_order_tool_tests() -> None:
    print("================================================================================")
    print("           NYKAA ASSIST TASK 21 — FIVE OPERATIONAL TOOLS VERIFICATION          ")
    print("================================================================================")

    res_status = check_order_status("NYK-00001")
    print(f"Status Tool (NYK-00001): {res_status['status']} | Value: {res_status['order_value_inr']}")
    assert res_status["status"] == "Placed"

    res_track = track_shipment("NYK-00006")
    print(f"Track Tool (NYK-00006): {res_track['status']} | Carrier: {res_track['carrier']} | Delayed: {res_track['delayed_shipment']}")
    assert res_track["delayed_shipment"] is True

    res_ret_elig = check_return_status("NYK-00004")
    print(f"Return Elig Tool (NYK-00004): Eligible={res_ret_elig['eligible']} | Category={res_ret_elig['category']}")
    assert res_ret_elig["eligible"] is True

    res_ret_exp = check_return_status("NYK-00002")
    print(f"Return Exp Tool (NYK-00002): Eligible={res_ret_exp['eligible']} | Category={res_ret_exp['category']}")
    assert res_ret_exp["eligible"] is False

    clear_return_requests()
    res_ret_create = create_return_request("NYK-00004", "Shade mismatch")
    print(f"Create Return Tool (NYK-00004): RMA={res_ret_create['rma_code']} | Status={res_ret_create['status']}")
    assert res_ret_create["status"] == "Initiated"

    res_ret_dup = create_return_request("NYK-00004", "Changed mind")
    assert res_ret_dup["rma_code"] == res_ret_create["rma_code"]
    print(f"Return Idempotency: Duplicate RMA matched existing RMA={res_ret_dup['rma_code']}")

    res_loyalty = loyalty_status("CUST-00002")
    print(f"Loyalty Tool (CUST-00002): Tier={res_loyalty['tier']} | Points={res_loyalty['points_balance']} | Spend={res_loyalty['lifetime_spend_inr']}")
    assert res_loyalty["tier"] == "Gold"
    assert res_loyalty["points_balance"] == 29

    print("================================================================================")
    print("      ALL FIVE OPERATIONAL TOOLS PASSED DIRECT VERIFICATION!                    ")
    print("================================================================================")


def main() -> None:
    run_order_tool_tests()


if __name__ == "__main__":
    main()
