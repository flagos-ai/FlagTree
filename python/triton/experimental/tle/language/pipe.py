# Copyright 2026- Xcoresigma Technology Co., Ltd

from triton.language.core import _unwrap_if_constexpr

from .dsa.core import builtin
from .dsa.ascend import pipe as ascend_pipe


def _pipe_backend(ready_sync, free_sync):
    # Fallback codegen paths may hand compile-time descriptors wrapped in constexpr.
    ready_sync = _unwrap_if_constexpr(ready_sync)
    free_sync = _unwrap_if_constexpr(free_sync)
    ready_backend = getattr(ready_sync, "backend", None)
    free_backend = getattr(free_sync, "backend", None)
    if ready_backend is None or free_backend is None:
        raise ValueError("ready_sync and free_sync must identify a pipe backend")
    if ready_backend != free_backend:
        raise ValueError("ready_sync and free_sync must use the same pipe backend")
    return ready_backend


@builtin
def pipe(*, capacity, scope="cta", name=None, ready_sync=None, free_sync=None, event_base=None, _semantic=None,
          _generator=None, **fields):
    if ready_sync is None or free_sync is None:
        raise ValueError("ready_sync and free_sync must be provided")
    backend = _pipe_backend(ready_sync, free_sync)
    if backend == "ascend":
        return ascend_pipe.pipe(capacity=capacity, scope=scope, name=name,
                                ready_sync=_unwrap_if_constexpr(ready_sync),
                                free_sync=_unwrap_if_constexpr(free_sync),
                                event_base=event_base, _semantic=_semantic,
                                _generator=_generator, **fields)
    raise ValueError(f"unsupported pipe backend: {backend!r}")
