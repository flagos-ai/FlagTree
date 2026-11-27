"""Guard: the pass namespaces the backend calls must resolve in the compiled
extension.

The backend builds its pipelines through `triton._C.libtriton`'s `passes`
namespaces; nothing else pins those names to what the C++ side registers.  A
wrong name is an AttributeError at *compile* time, on every kernel of that path
-- which is how the xpu3 checkin CI failed on every non-TLE kernel:

    AttributeError: module 'triton._C.libtriton.xpu.passes' has no attribute
    'ttgpuir'. Did you mean: 'ttxpuir'?

(`ttgpuir` is the core TritonGPU dialect's namespace; the XPU dialects register
under `ttxpuir`.)

This guard used to scan the backend's Python call sites for those names.  The
backend ships compiled now, so there is no text to scan -- it exercises the same
namespaces instead, by compiling one non-TLE kernel: a drifted name fails here,
on the path that actually consumes it.

Coverage is the branches that kernel reaches.  Names used only on TLE, on a
different `opt.arch`, or under non-default metadata are *not* covered by this
test; they surface at that path's first compile instead.

Run with:  pytest python/test/unit/backends/test_backend_pass_namespaces.py
"""

import pytest

triton = pytest.importorskip("triton")

import torch  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _probe_kernel(x_ptr, o_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) * 2.0)


def test_backend_pass_namespaces_resolve_when_compiling():
    # `warmup` compiles for the target without launching: the names this guards
    # are resolved at compile time, so there is nothing to run -- and a launch
    # would only add failure modes that have nothing to do with the namespaces.
    # It still needs a device (the target comes from one); the unit-test step
    # already runs with one.
    _probe_kernel.warmup(torch.float32, torch.float32, BLOCK=64, grid=(1, ))
