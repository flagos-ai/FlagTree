# Copyright 2026- Xcoresigma Technology Co., Ltd

from triton.language.core import _unwrap_if_constexpr

from .dsa.core import Workspace, builtin
from .dsa.ascend.pipe import (_deferred_pipe, pipe_reader, pipe_slot, pipe_value, pipe_wait_result, pipe_writer)


def _pipe_backend(fields):
    # GPU-aligned dispatch: the payload kind selects the backend, mirroring how
    # tle.gpu.buffered_tensor fields select the GPU backend on the NVIDIA path.
    for field in fields.values():
        field = _unwrap_if_constexpr(field)
        if not isinstance(field, Workspace):
            raise ValueError(f"tle.pipe field must be a tle.dsa.workspace payload, got {type(field).__name__}")
    return "ascend"


@builtin
def pipe(*, capacity, scope="cta", name=None, readers=None, one_shot=False, _semantic=None, **fields):
    backend = _pipe_backend(fields)
    if backend == "ascend":
        return _deferred_pipe(capacity=capacity, scope=scope, name=name, readers=readers, one_shot=one_shot,
                              _semantic=_semantic, **fields)
    raise ValueError(f"unsupported pipe backend: {backend!r}")


# GPU-shaped re-exports mirroring tle.language.pipe on the NVIDIA path.
__all__ = ["pipe", "pipe_value", "pipe_writer", "pipe_reader", "pipe_slot", "pipe_wait_result"]
