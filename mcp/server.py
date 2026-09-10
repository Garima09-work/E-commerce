import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastmcp import FastMCP
from agent.tools import (
    check_order_status as core_check_order_status,
    check_return_status as core_check_return_status,
    create_return_request as core_create_return_request,
    loyalty_status as core_loyalty_status,
    track_shipment as core_track_shipment,
)

ORDER_ID_PATTERN = re.compile(r"^NYK-\d{5}$", re.IGNORECASE)
CUSTOMER_ID_PATTERN = re.compile(r"^CUST-\d{5}$", re.IGNORECASE)

mcp_server = FastMCP("NykaaOrderService")


@mcp_server.tool(
    name="check_order_status",
    description="Lookup order status, order value in INR, and SLA escalation score for a valid Nykaa order record ID (NYK-XXXXX).",
)
def check_order_status(record_id: str) -> Dict[str, Any]:
    if not record_id or not isinstance(record_id, str):
        return {
            "record_id": str(record_id) if record_id is not None else "",
            "status": "Not Found",
            "order_value_inr": None,
            "escalation_score": None,
            "error": "Order ID must be a non-empty string.",
        }

    stripped_id = record_id.strip().upper()
    if not ORDER_ID_PATTERN.match(stripped_id):
        return {
            "record_id": stripped_id,
            "status": "Not Found",
            "order_value_inr": None,
            "escalation_score": None,
            "error": f"Invalid order ID format: '{record_id}'. Expected format is NYK-XXXXX.",
        }

    return core_check_order_status(stripped_id)


@mcp_server.tool(
    name="track_shipment",
    description="Retrieve live courier tracking details, current location, estimated delivery, and carrier delay alerts for an order ID (NYK-XXXXX).",
)
def track_shipment(record_id: str) -> Dict[str, Any]:
    if not record_id or not isinstance(record_id, str):
        return {
            "record_id": str(record_id) if record_id is not None else "",
            "status": "Invalid Identifier",
            "carrier": "N/A",
            "tracking_number": "N/A",
            "current_location": "N/A",
            "estimated_delivery": "N/A",
            "delayed_shipment": False,
            "error": "Order ID must be a non-empty string.",
        }

    stripped_id = record_id.strip().upper()
    if not ORDER_ID_PATTERN.match(stripped_id):
        return {
            "record_id": stripped_id,
            "status": "Invalid Identifier",
            "carrier": "N/A",
            "tracking_number": "N/A",
            "current_location": "N/A",
            "estimated_delivery": "N/A",
            "delayed_shipment": False,
            "error": f"Invalid order ID format: '{record_id}'. Expected format is NYK-XXXXX.",
        }

    return core_track_shipment(stripped_id)


@mcp_server.tool(
    name="check_return_status",
    description="Check whether an order is eligible for return based on delivery date and category policy window (15 days for Beauty/Apparel/Footwear, 7 days for Electronics/Home).",
)
def check_return_status(record_id: str) -> Dict[str, Any]:
    if not record_id or not isinstance(record_id, str):
        return {
            "record_id": str(record_id) if record_id is not None else "",
            "eligible": False,
            "category": None,
            "days_since_delivery": None,
            "allowed_window_days": None,
            "reason": "Order ID must be a non-empty string.",
            "error": "Order ID must be a non-empty string.",
        }

    stripped_id = record_id.strip().upper()
    if not ORDER_ID_PATTERN.match(stripped_id):
        return {
            "record_id": stripped_id,
            "eligible": False,
            "category": None,
            "days_since_delivery": None,
            "allowed_window_days": None,
            "reason": f"Invalid order ID format: '{record_id}'. Expected format is NYK-XXXXX.",
            "error": f"Invalid order ID format: '{record_id}'. Expected format is NYK-XXXXX.",
        }

    return core_check_return_status(stripped_id)


@mcp_server.tool(
    name="create_return_request",
    description="Initiate an authorized return request and generate an RMA code for an eligible delivered order. Idempotent and thread-safe.",
)
def create_return_request(record_id: str, reason: str) -> Dict[str, Any]:
    if not record_id or not isinstance(record_id, str):
        return {
            "record_id": str(record_id) if record_id is not None else "",
            "request_id": f"REQ-ERR-{record_id}",
            "status": "Failed",
            "rma_code": "N/A",
            "created_at": "",
            "reason": str(reason) if reason is not None else "",
            "error": "Order ID must be a non-empty string.",
        }

    stripped_id = record_id.strip().upper()
    if not ORDER_ID_PATTERN.match(stripped_id):
        return {
            "record_id": stripped_id,
            "request_id": f"REQ-ERR-{stripped_id}",
            "status": "Failed",
            "rma_code": "N/A",
            "created_at": "",
            "reason": str(reason) if reason is not None else "",
            "error": f"Invalid order ID format: '{record_id}'. Expected format is NYK-XXXXX.",
        }

    return core_create_return_request(stripped_id, str(reason))


@mcp_server.tool(
    name="loyalty_status",
    description="Retrieve customer loyalty rewards tier (Silver/Gold/Platinum), points balance, and lifetime spend for a customer ID (CUST-XXXXX).",
)
def loyalty_status(customer_id: str) -> Dict[str, Any]:
    if not customer_id or not isinstance(customer_id, str):
        return {
            "customer_id": str(customer_id) if customer_id is not None else "",
            "tier": "Silver",
            "points_balance": 0,
            "lifetime_spend_inr": 0.0,
            "error": "Customer ID must be a non-empty string.",
        }

    stripped_cid = customer_id.strip().upper()
    if not CUSTOMER_ID_PATTERN.match(stripped_cid):
        return {
            "customer_id": stripped_cid,
            "tier": "Silver",
            "points_balance": 0,
            "lifetime_spend_inr": 0.0,
            "error": f"Invalid customer ID format: '{customer_id}'. Expected format is CUST-XXXXX.",
        }

    return core_loyalty_status(stripped_cid)


def get_mcp_server() -> FastMCP:
    return mcp_server


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    transport: str = "sse",
) -> None:
    print(f"Starting NykaaAssist FastMCP Server on {host}:{port} (transport={transport})...")
    mcp_server.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    host_env = os.environ.get("MCP_HOST", "127.0.0.1")
    port_env = int(os.environ.get("MCP_PORT", "8000"))
    transport_env = os.environ.get("MCP_TRANSPORT", "sse")
    run_server(host=host_env, port=port_env, transport=transport_env)
