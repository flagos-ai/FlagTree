# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit diagnostic metadata for pipe boundaries; no runtime work here."""
import triton.language as tl
from triton.language.core import builtin


@builtin
def trace_configure(capacity, typed_ctas=None, _semantic=None):
    """Configure the separate native ring scratch ABI at compile time.

    Physical warp geometry and scratch argument ownership are derived by the
    native WS lowering. None selects runtime records without a typed CTA list.
    The host supplies aligned profile scratch and initializes its enable words.
    """
    capacity = tl.core._unwrap_if_constexpr(capacity)
    if type(capacity) is not int or not 1 <= capacity <= 1048576:
        raise ValueError('trace capacity must be a constant integer in [1, 1048576]')
    typed_ctas = tl.core._unwrap_if_constexpr(typed_ctas)
    if typed_ctas is not None:
        if not isinstance(typed_ctas, (list, tuple, tl.tuple)):
            raise ValueError('trace typed CTAs must be a static list or tuple')
        typed_ctas = [tl.core._unwrap_if_constexpr(cta) for cta in typed_ctas]
        if (not typed_ctas or any(type(cta) is not int or not 0 <= cta <= 0x7fffffff
                                  for cta in typed_ctas)
                or typed_ctas != sorted(set(typed_ctas))):
            raise ValueError('trace typed CTAs must be nonempty, strictly increasing nonnegative i32 IDs')
    if not hasattr(_semantic.builder, 'configure_tle_ws_trace'):
        raise ValueError('trace configuration requires a rebuilt NVIDIA TLE compiler')
    _semantic.builder.configure_tle_ws_trace(capacity, typed_ctas)


@builtin
def trace_event(kind, site, layer, iteration=-1, slot=-1, phase=-1, _semantic=None):
    """Observe an actual device-call boundary using the hidden profile ABI.

    Kinds 272/273 mean begin/end. A returned asynchronous device call does not
    imply that a later pipe drain or hardware copy has completed.
    """
    kind = tl.core._unwrap_if_constexpr(kind)
    site = tl.core._unwrap_if_constexpr(site)
    if type(kind) is not int or kind not in (272, 273):
        raise ValueError('module trace kind must be 272 (begin) or 273 (end)')
    if type(site) is not int or not 0 <= site <= 0x7fffffff:
        raise ValueError('module trace site must be a nonnegative signed i32 constant')
    values = []
    for name, value in (('layer', layer), ('iteration', iteration), ('slot', slot), ('phase', phase)):
        value = _semantic.to_tensor(value)
        if value.type.is_block() or value.dtype not in (tl.int32, tl.uint32):
            raise ValueError(f'module trace {name} must be a scalar i32; no implicit narrowing')
        values.append(value.handle)
    if not hasattr(_semantic.builder, 'create_tle_trace_event'):
        raise ValueError('module trace requires the NVIDIA TLE recorder backend')
    _semantic.builder.create_tle_trace_event(*values, kind, site)


def pipe_trace_identity(identity, iteration, trace, semantic):
    """trace=(nonnegative static site ID, scalar i32 runtime layer ID)."""
    trace = tl.core._unwrap_if_constexpr(trace)
    if trace is None:
        return identity
    if len(trace) != 2:
        raise ValueError('pipe trace requires (site_id, layer_id)')
    site = tl.core._unwrap_if_constexpr(trace[0])
    if not isinstance(site, int) or not 0 <= site <= 0x7fffffff:
        raise ValueError('pipe trace site ID must be a nonnegative signed i32 constant')
    layer = semantic.to_tensor(trace[1])
    iteration = semantic.to_tensor(iteration)
    for name, value in (('layer', layer), ('iteration', iteration)):
        if value.type.is_block() or value.dtype not in (tl.int32, tl.uint32):
            raise ValueError(f'pipe trace {name} must be a scalar i32; no implicit narrowing')
    if not hasattr(semantic.builder, 'create_pipe_trace_context'):
        raise ValueError('typed pipe trace requires the NVIDIA TLE recorder backend')
    return semantic.builder.create_pipe_trace_context(identity, layer.handle, iteration.handle, site)
