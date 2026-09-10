import json
import random
from pathlib import Path
from typing import Any, Dict, List

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.25, 0.15, 0.15, 0.15, 0.30]

STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.15, 0.25, 0.40, 0.10, 0.10]

PRICE_RANGES = {
    "Beauty": (299.0, 3499.0),
    "Apparel": (599.0, 4999.0),
    "Footwear": (899.0, 5999.0),
    "Electronics": (999.0, 11999.0),
    "Home": (499.0, 3999.0),
}

DELAY_PROBABILITY = 0.20
DEFAULT_SEED = 42
DEFAULT_COUNT = 50
DEFAULT_OUTPUT_PATH = "orders.json"


def generate_single_order(index: int) -> Dict[str, Any]:
    record_id = f"NYK-{index:05d}"
    category = random.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
    status = random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
    min_price, max_price = PRICE_RANGES[category]
    order_value = round(random.uniform(min_price, max_price), 2)
    days_since_created = random.randint(0, 30)
    delayed_shipment = random.random() < DELAY_PROBABILITY

    return {
        "record_id": record_id,
        "category": category,
        "status": status,
        "order_value_inr": order_value,
        "days_since_created": days_since_created,
        "delayed_shipment": delayed_shipment,
    }


def generate_dataset(seed: int = DEFAULT_SEED, count: int = DEFAULT_COUNT) -> List[Dict[str, Any]]:
    random.seed(seed)
    orders = [generate_single_order(i) for i in range(1, count + 1)]
    return orders


def save_dataset(records: List[Dict[str, Any]], filepath: str = DEFAULT_OUTPUT_PATH) -> None:
    path = Path(filepath)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def load_dataset(filepath: str = DEFAULT_OUTPUT_PATH) -> List[Dict[str, Any]]:
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def display_distributions(records: List[Dict[str, Any]]) -> None:
    total_count = len(records)
    category_counts = {category: 0 for category in CATEGORIES}
    status_counts = {status: 0 for status in STATUSES}
    delayed_count = 0

    for record in records:
        category_counts[record["category"]] += 1
        status_counts[record["status"]] += 1
        if record["delayed_shipment"]:
            delayed_count += 1

    delayed_percentage = (delayed_count / total_count) * 100 if total_count > 0 else 0.0

    print("========================================")
    print("      NYKAA ASSIST DATASET REPORT       ")
    print("========================================")
    print(f"Total Records Generated : {total_count}")
    print("\n--- Category Breakdown ---")
    for category, count in category_counts.items():
        percentage = (count / total_count) * 100
        print(f"  {category:<12} : {count:>2} ({percentage:5.1f}%)")

    print("\n--- Order Status Breakdown ---")
    for status, count in status_counts.items():
        percentage = (count / total_count) * 100
        print(f"  {status:<12} : {count:>2} ({percentage:5.1f}%)")

    print("\n--- Delivery Performance ---")
    print(f"  On-Time      : {total_count - delayed_count:>2} ({100.0 - delayed_percentage:5.1f}%)")
    print(f"  Delayed      : {delayed_count:>2} ({delayed_percentage:5.1f}%)")
    print("----------------------------------------")
    print(f"Delayed Shipment Rate : {delayed_percentage:.1f}% (Required: 10.0% - 30.0%)")
    if 10.0 <= delayed_percentage <= 30.0:
        print("Status               : VALID (Within SLA Threshold)")
    else:
        print("Status               : INVALID (Outside Required 10-30% Range)")
    print("========================================")


def main() -> None:
    dataset = generate_dataset()
    save_dataset(dataset)
    display_distributions(dataset)


if __name__ == "__main__":
    main()
