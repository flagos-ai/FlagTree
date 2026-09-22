"""End-to-end `tle.raw` test on the xpu3 cluster path: a native cluster-C vector add.

The payload (`vec_add.xpu`, next to this file) is XPU cluster C++ compiled with
`--xpu-arch=xpu3`, spliced into the kernel as an `always_inline llvm.call` during
`convert-tritonxpu-to-llvm`, and inlined into the cluster kernel by the LLVM
inliner.

These need a real XPU (KUNLUN3) and skip when no device is reachable; the
IR-level coverage that runs anywhere lives in ../test_tle_raw_cluster_ir.py.

What they pin down:
  - a payload taking global pointers + a scalar can be called from a Triton
    cluster kernel and its writes are visible to the host;
  - the result matches the equivalent pure-Triton kernel bit for bit (float32 is
    compared exactly, since the arithmetic is the same in the same order);
  - a tail that is not a multiple of the payload's tile is handled, and no
    element is left unwritten.

Note the kernel is launched WITHOUT a tl.dot: that is what selects the TritonSDNN
pipeline, where `sdnn.raw` lives. `triton_xpu.raw` only exists on the TritonXPU
(cluster) path, which is the default.
"""

import pytest

torch = pytest.importorskip("torch")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.experimental.tle as tle  # noqa: E402

# The payload strides over the range in TILE-sized (64) chunks, so cover: an
# exact multiple of the tile, a partial tail, a size below one tile, and a size
# that leaves some cores with no work at all.
SIZES = [1024, 1000, 64, 257]

BLOCK = 256


# "xpu3" is the XPU stack (third_party/xpu); the same handle serves the SDNN and
# the cluster path, the arch comes from the backend compiling the kernel, so run
# with TRITON_XPU_ARCH=3.
@tle.raw.dialect("xpu3", file="vec_add.xpu")
def vec_add(x, y, out, n):
    ...


@triton.jit
def vec_add_raw_kernel(X, Y, Out, n):
    # The payload owns the whole range: it strides over it with core_id() /
    # core_num() and stages each tile through LM, which is what the cluster
    # memory model requires.
    tle.raw.call(vec_add, (X, Y, Out, n))


@triton.jit
def vec_add_triton_kernel(X, Y, Out, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(Out + offs, x + y, mask=mask)


def _require_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("no XPU device available")
    # Name the ordinal: with XPU_FORCE_SHARED_DEVICE_CONTEXT both `.cuda()` and
    # `set_device` fail with "invalid device ordinal", while "cuda:0" works.
    return "cuda:0"


def _operands(n, dev):
    """Host operands, their device copies, and the host reference.

    Everything except the launch stays on the CPU: on the simulator a device-side
    `x + y` would dominate the test.
    """
    torch.manual_seed(0)
    x = torch.randn(n, dtype=torch.float32)
    y = torch.randn(n, dtype=torch.float32)
    ref = x + y
    # NaN-filled output: an element the payload never wrote stays NaN, so a short
    # write cannot pass by accident.
    out = torch.full((n, ), float("nan"), dtype=torch.float32)
    return x.to(dev), y.to(dev), out.to(dev), ref


@pytest.mark.parametrize("n", SIZES)
def test_raw_vector_add_matches_torch(n):
    dev = _require_device()
    x, y, out, ref = _operands(n, dev)

    vec_add_raw_kernel[(1, )](x, y, out, n)
    torch.cuda.synchronize()

    got = out.cpu()
    assert torch.isfinite(got).all(), "payload left part of the output unwritten"
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("n", SIZES)
def test_raw_vector_add_matches_triton(n):
    """Same arithmetic in the same order, so the two must agree exactly."""
    dev = _require_device()
    x, y, out_raw, _ = _operands(n, dev)
    out_tl = torch.full((n, ), float("nan"), dtype=torch.float32).to(dev)

    vec_add_raw_kernel[(1, )](x, y, out_raw, n)
    vec_add_triton_kernel[(triton.cdiv(n, BLOCK), )](x, y, out_tl, n, BLOCK_SIZE=BLOCK)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_raw.cpu(), out_tl.cpu(), rtol=0, atol=0)
