# precision.py — per-launch precision mode override for the tsingmicro backend.

# The effective precision mode must be decided by the launching thread: ops
# (flip/cat/copy/as_strided_copy) set it transiently around a kernel launch
# based on runtime tensor metadata. A process-global env var races under
# concurrent launches from multiple threads (one thread's restore could clobber
# another thread's set while the latter is mid-compile). The contextvar is
# per-context, so each thread's set/restore window is invisible to the others
# and needs no lock. The resolved value is frozen into TXDAOptions by
# parse_options, so the cache key and the compile pipeline both read the same
# per-kernel value.
import contextvars
import os

_precision_mode_cv: contextvars.ContextVar = contextvars.ContextVar("tsingmicro_precision_mode", default=None)


def set_precision_mode(value: str) -> contextvars.Token:
    if value not in ("0", "1", "2"):
        raise ValueError(f"precision mode must be 0, 1, or 2, got {value!r}")
    return _precision_mode_cv.set(value)


def restore_precision_mode(token: contextvars.Token) -> None:
    _precision_mode_cv.reset(token)


def get_precision_mode() -> str:
    # Explicit per-launch override, else the ambient env var (CI benchmark
    # scripts set PRECISION_MODE process-wide), else mode 0.
    value = _precision_mode_cv.get()
    if value is None:
        value = os.getenv("PRECISION_MODE", "0")
    value = value.strip()
    if value not in ("0", "1", "2"):
        raise ValueError("precision mode must be 0, 1, or 2")
    return value
