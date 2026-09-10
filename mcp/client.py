import asyncio
import importlib.util
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastmcp import Client

try:
    from server import CUSTOMER_ID_PATTERN, ORDER_ID_PATTERN, get_mcp_server
except Exception:
    _server_path = Path(__file__).resolve().parent / "server.py"
    _spec = importlib.util.spec_from_file_location("local_mcp_server", _server_path)
    _server_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_server_mod)
    ORDER_ID_PATTERN = _server_mod.ORDER_ID_PATTERN
    CUSTOMER_ID_PATTERN = _server_mod.CUSTOMER_ID_PATTERN
    get_mcp_server = _server_mod.get_mcp_server

DEFAULT_MCP_TOOL_NAME = "check_order_status"
APPROVED_MCP_TOOLS = {
    "check_order_status",
    "track_shipment",
    "check_return_status",
    "create_return_request",
    "loyalty_status",
}
MCP_CALL_HISTORY: List[Dict[str, Any]] = []


def get_mcp_call_count() -> int:
    return len(MCP_CALL_HISTORY)


def clear_mcp_call_history() -> None:
    MCP_CALL_HISTORY.clear()


async def _async_call_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    server_url: Optional[str] = None,
) -> Dict[str, Any]:
    if tool_name not in APPROVED_MCP_TOOLS:
        return {
            "error": f"Unauthorized tool name: '{tool_name}'. Allowed: {sorted(list(APPROVED_MCP_TOOLS))}",
        }

    MCP_CALL_HISTORY.append({
        "tool_name": tool_name,
        "arguments": arguments,
        "server_url": server_url,
    })
    target_url = server_url or os.environ.get("MCP_SERVER_URL")

    if target_url:
        try:
            async with Client(target_url) as client:
                result = await client.call_tool(tool_name, arguments)
                if hasattr(result, "data") and isinstance(result.data, dict):
                    return result.data
                if hasattr(result, "structured_content") and isinstance(result.structured_content, dict):
                    return result.structured_content
        except Exception:
            pass

    try:
        server_instance = get_mcp_server()
        async with Client(server_instance) as client:
            result = await client.call_tool(tool_name, arguments)
            if hasattr(result, "data") and isinstance(result.data, dict):
                return result.data
            if hasattr(result, "structured_content") and isinstance(result.structured_content, dict):
                return result.structured_content
    except Exception as exc:
        return {
            "error": f"MCP execution failed safely: {type(exc).__name__}",
        }

    return {
        "error": "Failed to parse MCP response",
    }


from resilience.retry_timeout import (
    NodeTimeoutError,
    ResilienceError,
    RetryExhaustedError,
    RetryPolicy,
    execute_with_retry,
    execute_with_timeout,
)

DEFAULT_MCP_TIMEOUT = 3.0
DEFAULT_MCP_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    initial_interval=0.05,
    backoff_factor=2.0,
    max_interval=1.0,
    jitter=0.0,
)


def _dispatch_raw_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    server_url: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(_async_call_mcp_tool(tool_name, arguments, server_url)))
            res = future.result()
    else:
        res = asyncio.run(_async_call_mcp_tool(tool_name, arguments, server_url))

    if isinstance(res, dict) and "error" in res and "status" not in res:
        err_msg = str(res.get("error", ""))
        transient_indicators = (
            "MCP execution failed safely",
            "Connection refused",
            "timed out",
            "timeout",
            "closed abruptly",
            "RemoteDisconnected",
            "ConnectionResetError",
        )
        if any(ind.lower() in err_msg.lower() for ind in transient_indicators):
            raise ResilienceError(err_msg)

    return res


def call_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    server_url: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    retry_policy: Optional[RetryPolicy] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    if tool_name not in APPROVED_MCP_TOOLS:
        return {
            "error": f"Unauthorized tool name: '{tool_name}'. Allowed: {sorted(list(APPROVED_MCP_TOOLS))}",
        }

    effective_timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_MCP_TIMEOUT
    active_policy = retry_policy or DEFAULT_MCP_RETRY_POLICY

    def _attempt_call() -> Dict[str, Any]:
        return execute_with_timeout(
            _dispatch_raw_mcp_tool,
            tool_name,
            arguments,
            server_url,
            timeout_seconds=effective_timeout,
            trace_id=trace_id,
            error_cls=NodeTimeoutError,
        )

    try:
        return execute_with_retry(
            _attempt_call,
            policy=active_policy,
            trace_id=trace_id,
        )
    except NodeTimeoutError as t_err:
        return {
            "error": f"MCP tool '{tool_name}' timed out after {effective_timeout}s: {t_err}",
            "status": "Timeout",
        }
    except RetryExhaustedError as r_err:
        is_timeout = isinstance(r_err.__cause__, (NodeTimeoutError, TimeoutError))
        return {
            "error": f"MCP tool '{tool_name}' failed after {active_policy.max_attempts} attempts: {r_err}",
            "status": "Timeout" if is_timeout else "Retry Exhausted",
        }
    except Exception as exc:
        return {
            "error": f"MCP execution failed safely: {type(exc).__name__}: {exc}",
            "status": "Error",
        }


def call_order_tool_mcp(
    record_id: str,
    server_url: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    retry_policy: Optional[RetryPolicy] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
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

    effective_timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_MCP_TIMEOUT
    res = call_mcp_tool(
        DEFAULT_MCP_TOOL_NAME,
        {"record_id": stripped_id},
        server_url,
        timeout_seconds=effective_timeout,
        retry_policy=retry_policy,
        trace_id=trace_id,
    )
    if "error" in res and ("status" not in res or res.get("status") in ("Timeout", "Retry Exhausted", "Error")):
        return {
            "record_id": stripped_id,
            "status": "Not Found",
            "order_value_inr": None,
            "escalation_score": None,
            "error": res["error"],
        }
    return res


async def discover_mcp_tools(server_url: Optional[str] = None) -> List[str]:
    target_url = server_url or os.environ.get("MCP_SERVER_URL")
    if target_url:
        try:
            async with Client(target_url) as client:
                tools = await client.list_tools()
                return [t.name for t in tools]
        except Exception:
            pass

    server_instance = get_mcp_server()
    async with Client(server_instance) as client:
        tools = await client.list_tools()
        return [t.name for t in tools]


def run_standalone_client_tests() -> None:
    print("================================================================================")
    print("           NYKAA ASSIST TASK 21 — STANDALONE MCP CLIENT VERIFICATION           ")
    print("================================================================================")

    tool_names = asyncio.run(discover_mcp_tools())
    print(f"Discovered MCP Tools : {tool_names}")
    expected_tools = sorted(list(APPROVED_MCP_TOOLS))
    assert sorted(tool_names) == expected_tools, f"Expected {expected_tools}, got {sorted(tool_names)}"
    print(f"[PASSED] Tool isolation verified: Exactly 5 operational tools exposed: {expected_tools}!")

    test_cases = [
        ("NYK-00001", "Placed"),
        ("NYK-00007", "Returned"),
        ("NYK-99999", "Not Found"),
    ]

    for order_id, expected_status in test_cases:
        res = call_order_tool_mcp(order_id)
        print(f"\nMCP Tool Call: '{order_id}'")
        print(f"  Record ID        : {res.get('record_id')}")
        print(f"  Status           : {res.get('status')}")
        print(f"  Order Value (INR): {res.get('order_value_inr')}")
        print(f"  Escalation Score : {res.get('escalation_score')}")
        if res.get("error"):
            print(f"  Error Detail     : {res.get('error')}")

        assert res.get("record_id") == order_id
        assert res.get("status") == expected_status
        print(f"  [PASSED] Round trip validated for {order_id} (Status: {expected_status})")

    res_track = call_mcp_tool("track_shipment", {"record_id": "NYK-00006"})
    assert res_track.get("delayed_shipment") is True
    print(f"[PASSED] track_shipment via call_mcp_tool verified (Delayed={res_track.get('delayed_shipment')})")

    res_ret_elig = call_mcp_tool("check_return_status", {"record_id": "NYK-00004"})
    assert res_ret_elig.get("eligible") is True
    print(f"[PASSED] check_return_status via call_mcp_tool verified (Eligible={res_ret_elig.get('eligible')})")

    res_ret_req = call_mcp_tool("create_return_request", {"record_id": "NYK-00004", "reason": "Shade too dark"})
    assert res_ret_req.get("status") == "Initiated"
    assert "RMA-NYK-00004" in res_ret_req.get("rma_code", "")
    print(f"[PASSED] create_return_request via call_mcp_tool verified (RMA={res_ret_req.get('rma_code')})")

    res_loyalty = call_mcp_tool("loyalty_status", {"customer_id": "CUST-00002"})
    assert res_loyalty.get("tier") == "Gold"
    assert res_loyalty.get("points_balance") == 29
    print(f"[PASSED] loyalty_status via call_mcp_tool verified (Tier={res_loyalty.get('tier')}, Points={res_loyalty.get('points_balance')})")

    unauthorized_res = call_mcp_tool("delete_database", {})
    assert "error" in unauthorized_res
    print(f"[PASSED] Unauthorized tool call blocked safely: {unauthorized_res.get('error')}")

    malformed_cases = [
        "INVALID-123",
        "../orders.json",
        "NYK-01",
    ]
    for malformed_id in malformed_cases:
        res = call_order_tool_mcp(malformed_id)
        assert res.get("status") == "Not Found"
        assert "error" in res
        print(f"[PASSED] Malformed identifier '{malformed_id}' safely rejected before tool execution.")

    print("\n================================================================================")
    print("      NYKAA ASSIST TASK 21 — ALL STANDALONE MCP CLIENT TESTS PASSED!             ")
    print("================================================================================")


try:
    import mcp.client as _sdk_client_pkg
    setattr(_sdk_client_pkg, "call_order_tool_mcp", call_order_tool_mcp)
    setattr(_sdk_client_pkg, "call_mcp_tool", call_mcp_tool)
    setattr(_sdk_client_pkg, "discover_mcp_tools", discover_mcp_tools)
    setattr(_sdk_client_pkg, "get_mcp_call_count", get_mcp_call_count)
    setattr(_sdk_client_pkg, "clear_mcp_call_history", clear_mcp_call_history)
    setattr(_sdk_client_pkg, "DEFAULT_MCP_TIMEOUT", DEFAULT_MCP_TIMEOUT)
    setattr(_sdk_client_pkg, "DEFAULT_MCP_RETRY_POLICY", DEFAULT_MCP_RETRY_POLICY)
except Exception:
    pass


if __name__ == "__main__":
    run_standalone_client_tests()
