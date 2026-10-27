"""Thread-safe run-level limits shared by parallel claim workers."""

from __future__ import annotations

import threading
import time


# A request is not started (and a retry sleep is not begun) unless at least this
# much run time would remain for the request itself.
MIN_REQUEST_S = 5.0
CONNECT_TIMEOUT_CAP_S = 10.0


def split_timeout(total_s: float) -> tuple[float, float]:
    """Split one per-request budget into (connect, read) phase limits.

    httpx and requests apply a scalar timeout to *each* phase separately
    (connect incl. TLS handshake, then each read), so a scalar of 60 s allows
    ~60 s to connect plus another ~60 s waiting for the response. Splitting
    makes connection stalls fail fast and makes connect + read == total.
    """
    connect = min(CONNECT_TIMEOUT_CAP_S, total_s / 4.0)
    return connect, total_s - connect


class BudgetExceeded(RuntimeError):
    """A per-run budget (LLM calls or search credits) would be exceeded."""


class Deadline:
    def __init__(self, seconds: float, clock=time.monotonic):
        self._clock = clock
        self._end = clock() + seconds
        self.seconds = seconds

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


class CallBudget:
    """Counts logical LLM calls; refuses once ``limit`` is reached."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def acquire(self, n: int = 1) -> None:
        with self._lock:
            if self.used + n > self.limit:
                raise BudgetExceeded(f"LLM call budget of {self.limit} reached")
            self.used += n


class CreditBudget:
    """Search-credit budget. Credits are reserved *before* a request is sent,
    so the configured budget is never exceeded even under concurrency."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def reserve(self, n: int) -> bool:
        with self._lock:
            if self.used + n > self.limit:
                return False
            self.used += n
            return True

    def refund(self, n: int) -> None:
        with self._lock:
            self.used = max(0, self.used - n)
