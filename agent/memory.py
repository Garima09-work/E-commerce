import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.guardrails import detect_prompt_injection, mask_pii

DEFAULT_DB_PATH = "checkpoints.sqlite"


class GuardrailSqliteSaver(SqliteSaver):
    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)
        self.blocked_threads = set()

    def _extract_query_text(self, checkpoint: Any) -> str:
        channel_values = checkpoint.get("channel_values", {})
        extracted_parts = []
        if isinstance(channel_values, dict):
            for key in ("query", "original_query", "rewritten_query"):
                val = channel_values.get(key)
                if isinstance(val, str):
                    extracted_parts.append(val)
            start_val = channel_values.get("__start__")
            if isinstance(start_val, dict):
                for key in ("query", "original_query", "rewritten_query"):
                    start_query = start_val.get(key)
                    if isinstance(start_query, str):
                        extracted_parts.append(start_query)
        return " ".join(extracted_parts).strip()

    def _sanitize_data(self, data: Any) -> Any:
        if isinstance(data, str):
            return mask_pii(data)
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_data(v) for v in data]
        return data

    def put(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any:
        thread_id = str(config["configurable"]["thread_id"])
        query_text = self._extract_query_text(checkpoint)
        channel_values = checkpoint.get("channel_values", {})
        is_blocked = (
            channel_values.get("is_blocked")
            if isinstance(channel_values, dict)
            else False
        )

        flagged, _ = detect_prompt_injection(query_text)
        if flagged or is_blocked:
            self.blocked_threads.add(thread_id)
            checkpoint_ns = config["configurable"]["checkpoint_ns"]
            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                }
            }

        self.blocked_threads.discard(thread_id)
        if "channel_values" in checkpoint:
            checkpoint["channel_values"] = self._sanitize_data(
                checkpoint["channel_values"]
            )
        return super().put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: Any,
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = str(config["configurable"]["thread_id"])
        if thread_id in self.blocked_threads:
            return
        sanitized_writes = [
            (channel, self._sanitize_data(val)) for channel, val in writes
        ]
        super().put_writes(config, sanitized_writes, task_id, task_path)


def get_sqlite_checkpointer(
    db_path: str = DEFAULT_DB_PATH,
) -> GuardrailSqliteSaver:
    target_path = Path(db_path)
    if not target_path.is_absolute():
        target_path = ROOT_DIR / target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=30.0, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass
    saver = GuardrailSqliteSaver(conn)
    saver.setup()
    return saver


def get_thread_history(
    thread_id: str,
    checkpointer: Optional[GuardrailSqliteSaver] = None,
) -> List[Dict[str, Any]]:
    active_saver = checkpointer or get_sqlite_checkpointer()
    config = {"configurable": {"thread_id": str(thread_id)}}
    checkpoints = list(active_saver.list(config))
    history = []
    for cp in checkpoints:
        channel_vals = cp.checkpoint.get("channel_values", {})
        history.append(channel_vals)
    return history


def clear_thread_checkpoints(
    thread_id: str,
    checkpointer: Optional[GuardrailSqliteSaver] = None,
) -> None:
    active_saver = checkpointer or get_sqlite_checkpointer()
    active_saver.delete_thread(str(thread_id))


def verify_no_raw_pii_in_db(
    db_path: str = DEFAULT_DB_PATH,
    sensitive_strings: Optional[List[str]] = None,
) -> bool:
    target_path = Path(db_path)
    if not target_path.is_absolute():
        target_path = ROOT_DIR / target_path

    if not target_path.exists():
        return True

    if sensitive_strings is None:
        sensitive_strings = []

    conn = sqlite3.connect(str(target_path))
    cursor = conn.cursor()

    cursor.execute("SELECT checkpoint FROM checkpoints")
    for (row_data,) in cursor.fetchall():
        text_data = str(row_data)
        for s in sensitive_strings:
            if s and s in text_data:
                conn.close()
                return False

    cursor.execute("SELECT value FROM writes")
    for (row_data,) in cursor.fetchall():
        text_data = str(row_data)
        for s in sensitive_strings:
            if s and s in text_data:
                conn.close()
                return False

    conn.close()
    return True
