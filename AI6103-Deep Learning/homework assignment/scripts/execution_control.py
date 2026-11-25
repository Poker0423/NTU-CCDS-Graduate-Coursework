"""Cooperative stop at minibatch boundaries; the last committed epoch stays intact."""
import os
import signal
import time

_stop = None
_deadline_monotonic = None


class TrainingStopped(RuntimeError):
    pass


def install_handlers():
    global _stop, _deadline_monotonic
    _stop = None
    deadline = os.environ.get("AI6103_SOFT_DEADLINE_EPOCH")
    _deadline_monotonic = time.monotonic() + max(0, float(deadline) - time.time()) if deadline else None
    def handler(signum, frame):
        global _stop
        _stop = f"signal {signum}"
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def check_stop():
    if _stop:
        raise TrainingStopped(_stop)
    if _deadline_monotonic is not None and time.monotonic() >= _deadline_monotonic:
        raise TrainingStopped("Wall-clock budget approaching; last committed epoch preserved")
