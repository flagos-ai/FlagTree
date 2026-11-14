"""The `disable_load_vectorize` compile option emits scalar global loads.

Wide global loads on BI-V150 are bound by load-instruction issue rate (roughly
one per cycle per MP) rather than by HBM bandwidth, so a read-bound kernel at
high occupancy can be markedly faster with scalar loads, while a kernel pinned
to one CTA per SM is faster with wide ones. The crossover depends on
CTAs-per-SM, which the lowering cannot see, so this is an explicit opt-out
rather than a heuristic.

Set it like any other Corex compile option -- as a launch kwarg, or through
`options=` on `triton.compile`:

    kernel[grid](x, out, BLOCK_SIZE=1024, disable_load_vectorize=True)

These tests pin the contract: the option reaches the lowering, it changes only
the LLVM load width, it leaves the pre-LLVM IR untouched, stores keep their
width, results are unchanged, and the two variants cache separately.

Kernels are compiled through a real launch where load width matters, because
vectorization needs the `tt.divisibility` argument specialization that only an
actual launch supplies; a bare `ASTSource` yields scalar loads even by default
and would make those assertions vacuous.
"""

import re

import pytest
import torch

import triton
import triton.language as tl

BLOCK = 1024

# Global accesses lower to llvm.bi.{load,store}.kop.<vecty> intrinsics, e.g.
# `call <4 x float> @llvm.bi.load.kop.v4f32(...)`; scalar accesses are v1f32.
_LOAD_RE = re.compile(r"@llvm\.bi\.load\.kop\.v(\d+)[a-z]\d+")
_STORE_RE = re.compile(r"@llvm\.bi\.store\.kop\.v(\d+)[a-z]\d+")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


@triton.jit
def _copy(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))


@triton.jit
def _copy_masked(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def _run(kernel, disable, masked=False):
    """Launch with/without the option, check numerics, return the artifact."""
    kernel.device_caches.clear()
    x = torch.randn(BLOCK * 4, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    kwargs = {"BLOCK_SIZE": BLOCK, "disable_load_vectorize": disable}
    if masked:
        kernel[(1, )](x, out, BLOCK, **kwargs)
    else:
        kernel[(1, )](x, out, **kwargs)
    torch.testing.assert_close(out[:BLOCK], x[:BLOCK])
    compiled = [c for _, entry in kernel.device_caches.items() for _, c in entry[0].items()]
    assert len(compiled) == 1, f"expected one compiled variant, got {len(compiled)}"
    return compiled[0]


def _load_widths(compiled):
    return [int(n) for n in _LOAD_RE.findall(compiled.asm["llir"])]


def _store_widths(compiled):
    return [int(n) for n in _STORE_RE.findall(compiled.asm["llir"])]


def test_option_is_accepted_and_recorded():
    """It must be a real Corex option, not silently swallowed."""
    from triton.backends.iluvatar.compiler import CorexOptions
    assert "disable_load_vectorize" in CorexOptions.__dataclass_fields__
    assert CorexOptions().disable_load_vectorize is False
    assert _run(_copy, True).metadata.disable_load_vectorize is True
    assert _run(_copy, False).metadata.disable_load_vectorize is False


def test_unknown_option_still_rejected():
    """Guard against the option plumbing accepting arbitrary kwargs."""
    x = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    with pytest.raises(KeyError):
        _copy[(1, )](x, out, BLOCK_SIZE=BLOCK, not_a_real_option=True)


def test_loads_become_scalar():
    """The option must actually remove vectorized global loads."""
    baseline = _load_widths(_run(_copy, False))
    narrowed = _load_widths(_run(_copy, True))
    assert max(baseline) > 1, f"expected vectorized loads by default, got {baseline}"
    assert set(narrowed) == {1}, f"expected only scalar loads, got {narrowed}"


def test_masked_loads_become_scalar():
    """The mask path routes through getMaskElemsAndUpdateVeclen; cover it too."""
    baseline = _load_widths(_run(_copy_masked, False, masked=True))
    narrowed = _load_widths(_run(_copy_masked, True, masked=True))
    assert max(baseline) > 1, f"expected vectorized loads by default, got {baseline}"
    assert set(narrowed) == {1}, f"expected only scalar loads, got {narrowed}"


def test_stores_keep_their_width():
    """Store bandwidth is width-insensitive here, so stores must be unaffected."""
    baseline = _store_widths(_run(_copy, False))
    narrowed = _store_widths(_run(_copy, True))
    assert baseline == narrowed, f"store widths changed: {baseline} -> {narrowed}"


def test_pre_llvm_ir_is_identical():
    """Lowering-only: every stage before lower-to-LLVM must be byte-identical.

    This is what proves contiguity/AxisInfo analysis and all earlier passes are
    untouched -- unlike a tt.func attribute, a pass option leaves no trace in
    the incoming IR at all.
    """
    baseline = _run(_copy, False)
    narrowed = _run(_copy, True)
    for stage in ("ttir", "ttgir"):
        assert baseline.asm[stage] == narrowed.asm[stage], f"{stage} differs"


def test_variants_cache_separately():
    """A flipped option must not reuse the previously compiled kernel."""
    _copy.device_caches.clear()
    x = torch.randn(BLOCK, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    _copy[(1, )](x, out, BLOCK_SIZE=BLOCK, disable_load_vectorize=False)
    _copy[(1, )](x, out, BLOCK_SIZE=BLOCK, disable_load_vectorize=True)
    compiled = [c for _, entry in _copy.device_caches.items() for _, c in entry[0].items()]
    assert len(compiled) == 2, f"expected two cached variants, got {len(compiled)}"
    widths = sorted(set(w for c in compiled for w in _load_widths(c)))
    assert widths == [1, 4], f"expected both scalar and 4-wide variants, got {widths}"
