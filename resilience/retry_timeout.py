import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

T = TypeVar("T")


class ResilienceError(Exception):
    pass


class NodeTimeoutError(TimeoutError, ResilienceError):
    def __init__(self, message: str, node_name: Optional[str] = None, timeout_seconds: Optional[float] = None):
        super().__init__(message)
        self.node_name = node_name
        self.timeout_seconds = timeout_seconds


class GlobalTimeoutError(TimeoutError, ResilienceError):
    def __init__(self, message: str, timeout_seconds: Optional[float] = None):
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class RetryExhaustedError(ResilienceError):
    def __init__(self, message: str, attempts: int = 0):
        super().__init__(message)
        self.attempts = attempts


class ResilienceEvent(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            return None


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        initial_interval: float = 0.05,
        backoff_factor: float = 2.0,
        max_interval: float = 1.0,
        jitter: Any = 0.0,
        retryable_exceptions: Optional[Tuple[Type[BaseException], ...]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if initial_interval < 0.0:
            raise ValueError("initial_interval cannot be negative")
        if backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")
        if max_interval < initial_interval:
            raise ValueError("max_interval cannot be less than initial_interval")

        self.max_attempts = max_attempts
        self.initial_interval = initial_interval
        self.backoff_factor = backoff_factor
        self.max_interval = max_interval
        self.seed = seed

        if isinstance(jitter, bool):
            self.jitter = 1.0 if jitter else 0.0
            self.jitter_is_bool = jitter
        elif isinstance(jitter, (int, float)):
            if jitter < 0.0:
                raise ValueError("jitter cannot be negative")
            self.jitter = float(jitter)
            self.jitter_is_bool = False
        else:
            self.jitter = 0.0
            self.jitter_is_bool = False

        self.retryable_exceptions = retryable_exceptions or (
            TimeoutError,
            ConnectionError,
            OSError,
            IOError,
            ResilienceError,
        )

    def calculate_delay(self, attempt: int, seed: Optional[int] = None) -> float:
        if attempt <= 1:
            base_delay = self.initial_interval
        else:
            base_delay = self.initial_interval * (self.backoff_factor ** (attempt - 1))

        clamped_delay = min(base_delay, self.max_interval)

        effective_seed = seed if seed is not None else self.seed
        if self.jitter > 0.0:
            if effective_seed is not None:
                pseudo_rand = (((effective_seed + attempt * 17) * 9301 + 49297) % 233280) / 233280.0
                if self.jitter_is_bool:
                    return pseudo_rand * clamped_delay
                else:
                    jitter_val = pseudo_rand * self.jitter
                    return min(clamped_delay + jitter_val, self.max_interval)
            else:
                if self.jitter_is_bool:
                    return random.uniform(0.0, clamped_delay)
                else:
                    jitter_val = random.uniform(0.0, self.jitter)
                    return min(clamped_delay + jitter_val, self.max_interval)

        return clamped_delay

    def get_interval(self, attempt: int) -> float:
        return self.calculate_delay(attempt, seed=self.seed)


RESILIENCE_EVENTS: List[ResilienceEvent] = []
RESILIENCE_LOCK = threading.Lock()


def record_resilience_event(event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
    entry = ResilienceEvent({
        "event_type": event_type,
        "timestamp": time.time(),
    })
    if details:
        entry.update(details)
    with RESILIENCE_LOCK:
        RESILIENCE_EVENTS.append(entry)


def get_resilience_events() -> List[ResilienceEvent]:
    with RESILIENCE_LOCK:
        return list(RESILIENCE_EVENTS)


def clear_resilience_events() -> None:
    with RESILIENCE_LOCK:
        RESILIENCE_EVENTS.clear()


def execute_with_retry(
    callable_fn: Callable[..., T],
    *args: Any,
    policy: Optional[RetryPolicy] = None,
    trace_id: Optional[str] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> T:
    active_policy = policy or RetryPolicy()

    for attempt in range(1, active_policy.max_attempts + 1):
        try:
            result = callable_fn(*args, **kwargs)
            if attempt > 1:
                record_resilience_event(
                    "retry_success",
                    {
                        "attempt": attempt,
                        "trace_id": trace_id,
                        "function": getattr(callable_fn, "__name__", str(callable_fn)),
                    },
                )
                record_resilience_event(
                    "retry_recovered",
                    {
                        "attempt": attempt,
                        "trace_id": trace_id,
                        "function": getattr(callable_fn, "__name__", str(callable_fn)),
                    },
                )
            return result
        except Exception as exc:
            if not isinstance(exc, active_policy.retryable_exceptions):
                record_resilience_event(
                    "non_retryable_error",
                    {
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "trace_id": trace_id,
                    },
                )
                raise exc

            if attempt >= active_policy.max_attempts:
                record_resilience_event(
                    "retry_exhausted",
                    {
                        "attempts_executed": attempt,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "trace_id": trace_id,
                    },
                )
                raise RetryExhaustedError(
                    f"Execution exhausted all {active_policy.max_attempts} attempts: {exc}",
                    attempts=attempt,
                ) from exc

            delay = active_policy.calculate_delay(attempt, seed=seed)
            record_resilience_event(
                "retry_attempt",
                {
                    "attempt": attempt,
                    "delay_seconds": delay,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "trace_id": trace_id,
                },
            )
            record_resilience_event(
                "retry_backoff",
                {
                    "attempt": attempt,
                    "delay_seconds": delay,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "trace_id": trace_id,
                },
            )
            if on_retry:
                on_retry(attempt, exc, delay)
            time.sleep(delay)

    raise RetryExhaustedError("Retry loop completed without result or captured exception", attempts=active_policy.max_attempts)


def execute_with_timeout(
    callable_fn: Callable[..., T],
    *args: Any,
    timeout_seconds: float = 3.0,
    trace_id: Optional[str] = None,
    node_name: Optional[str] = None,
    error_cls: Type[TimeoutError] = NodeTimeoutError,
    **kwargs: Any,
) -> T:
    if timeout_seconds <= 0.0:
        return callable_fn(*args, **kwargs)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(callable_fn, *args, **kwargs)

    try:
        result = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=False, cancel_futures=True)
        return result
    except FutureTimeoutError as f_err:
        executor.shutdown(wait=False, cancel_futures=True)
        record_resilience_event(
            "timeout_triggered",
            {
                "timeout_seconds": timeout_seconds,
                "node_name": node_name,
                "error_cls": error_cls.__name__,
                "trace_id": trace_id,
                "function": getattr(callable_fn, "__name__", str(callable_fn)),
            },
        )
        if issubclass(error_cls, NodeTimeoutError):
            err_inst = error_cls(
                f"Operation in node '{node_name}' timed out after {timeout_seconds} seconds",
                node_name=node_name,
                timeout_seconds=timeout_seconds,
            )
        elif issubclass(error_cls, GlobalTimeoutError):
            err_inst = error_cls(
                f"Global graph operation timed out after {timeout_seconds} seconds",
                timeout_seconds=timeout_seconds,
            )
        else:
            err_inst = error_cls(f"Operation timed out after {timeout_seconds} seconds")
        raise err_inst from f_err
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise


class GlobalTimeoutGuard:
    def __init__(
        self,
        timeout_seconds: float = 15.0,
        trace_id: Optional[str] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.trace_id = trace_id
        self.start_time: float = 0.0

    def __enter__(self) -> "GlobalTimeoutGuard":
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        pass

    def check_deadline(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            record_resilience_event(
                "global_timeout_overrun",
                {
                    "elapsed_seconds": elapsed,
                    "deadline_seconds": self.timeout_seconds,
                    "trace_id": self.trace_id,
                },
            )
            raise GlobalTimeoutError(
                f"Global run deadline exceeded: elapsed {elapsed:.3f}s > {self.timeout_seconds:.3f}s"
            )


class SimulatedFlakyOperation:
    def __init__(
        self,
        fail_count: int = 2,
        error_type: Type[Exception] = ConnectionError,
        success_value: Any = "success",
        sleep_duration: float = 0.0,
        error_message: str = "Simulated transient network drop",
        failure_exception: Optional[Exception] = None,
    ) -> None:
        self.fail_count = fail_count
        self.error_type = error_type
        self.success_value = success_value
        self.sleep_duration = sleep_duration
        self.error_message = error_message
        self.failure_exception = failure_exception
        self.call_count = 0
        self.lock = threading.Lock()

    @property
    def attempts(self) -> int:
        return self.call_count

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self.lock:
            self.call_count += 1
            current_call = self.call_count

        if current_call <= self.fail_count:
            if self.failure_exception is not None:
                raise self.failure_exception
            raise self.error_type(f"{self.error_message} (attempt {current_call}/{self.fail_count})")

        if self.sleep_duration > 0.0:
            time.sleep(self.sleep_duration)

        return self.success_value

    def reset(self) -> None:
        with self.lock:
            self.call_count = 0
