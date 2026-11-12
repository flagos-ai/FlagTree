"""End-to-end `tle.raw` smoke on xpu3 via a prebuilt object payload.

Same layout as `../vector_add/test_tle_raw_vector_add.py`, but uses
`object=reflection_pad1d.o` (FlagGems #6767) so the public desensitized
toolchain does not need `llvm-link` to merge `.xpu` IR text.

The payload only writes the reflected pad edges; the interior is host-copied
first. Needs a real XPU (KUNLUN3); skips when none is reachable.
"""

import os

import pytest

torch = pytest.importorskip("torch")
import triton.language as tl  # noqa: E402
import triton.experimental.tle as tle  # noqa: E402

_PAYLOAD = os.path.join(os.path.dirname(__file__), "reflection_pad1d.o")

# One giant-ish shape so the edge kernel does real work (B * pad covers lanes).
B, W_IN, PAD_L, PAD_R = 32, 256, 3, 5
W_OUT = W_IN + PAD_L + PAD_R


@tle.raw.dialect("xpu3", object=_PAYLOAD, arch=3)
def pad_edges3(out, inp, B, W_in, W_out, pad_left, pad_right, es, pid, npid):
    ...


@triton.jit(do_not_specialize=["B", "W_in", "W_out", "pad_left", "pad_right", "es", "npid"])
def pad1d_raw_edges_kernel(out_i8, in_i8, B, W_in, W_out, pad_left, pad_right, es, npid):
    pid = tl.program_id(axis=0)
    tle.raw.call(
        pad_edges3,
        (out_i8, in_i8, B, W_in, W_out, pad_left, pad_right, es, pid, npid),
    )


def _require_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("no XPU device available")
    return "cuda:0"


def test_raw_reflection_pad1d_matches_torch():
    dev = _require_device()

    torch.manual_seed(0)
    x = torch.randn(B, W_IN, dtype=torch.float32, device=dev)
    out = torch.empty(B, W_OUT, dtype=x.dtype, device=dev)
    # Interior: payload does not touch it.
    out[:, PAD_L:PAD_L + W_IN].copy_(x)

    npid = min(32, B)
    pad1d_raw_edges_kernel[(npid, )](
        out.view(torch.int8),
        x.view(torch.int8),
        B,
        W_IN,
        W_OUT,
        PAD_L,
        PAD_R,
        x.element_size(),
        npid,
    )
    torch.cuda.synchronize()

    ref = torch.nn.functional.pad(x, (PAD_L, PAD_R), mode="reflect")
    torch.testing.assert_close(out.cpu(), ref.cpu())
