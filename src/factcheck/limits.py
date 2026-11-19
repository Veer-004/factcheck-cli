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
    so the configured budget is never exceeded even under concurrency.

    Fair sharing: once the run knows its checkable claims, ``allot`` gives every
    claim an equal guaranteed share (the remainder goes one credit each to the
    first claims). A claim spends its own share first. It can draw on the shared
    pool, which holds only the credits other claims released after their first
    search round (``release``). A fast claim's second round therefore can never
    use credits that a slower claim has not had the chance to spend. Before
    ``allot`` (or for requests without a claim id) the budget is one shared pool.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()
        self._shares: dict[str, int] | None = None
        self._pool = 0

    def allot(self, claim_ids: list[str]) -> dict[str, int]:
        with self._lock:
            free = self.limit - self.used
            if not claim_ids:
                self._shares = None
                return {}
            base, extra = divmod(free, len(claim_ids))
            self._shares = {cid: base + (1 if i < extra else 0) for i, cid in enumerate(claim_ids)}
            self._pool = 0
            return dict(self._shares)

    def release(self, claim_id: str) -> None:
        """Hand the claim's unspent share to the shared pool (idempotent)."""
        with self._lock:
            if self._shares and self._shares.get(claim_id):
                self._pool += self._shares[claim_id]
                self._shares[claim_id] = 0

    def reserve(self, n: int, claim_id: str | None = None) -> bool:
        with self._lock:
            if self.used + n > self.limit:
                return False
            if self._shares is not None and claim_id is not None:
                own = min(self._shares.get(claim_id, 0), n)
                if n - own > self._pool:
                    return False
                self._shares[claim_id] = self._shares.get(claim_id, 0) - own
                self._pool -= n - own
            self.used += n
            return True
