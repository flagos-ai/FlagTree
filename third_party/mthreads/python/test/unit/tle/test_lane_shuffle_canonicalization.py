"""Prove a physical XOR lane map; keep arbitrary dynamic gathers unchanged."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _lane_gather(x, y, delta, L: tl.constexpr, MASK: tl.constexpr, DYNAMIC: tl.constexpr):
    m, n = tl.arange(0, 4), tl.arange(0, 32)
    value = tle.gpu.set_layout(tl.load(x + m[:, None] * 32 + n[None, :]), L)
    index = (n + delta) & 31 if DYNAMIC else n ^ MASK
    index = tle.gpu.set_layout(tl.broadcast_to(index[None, :], (4, 32)), L)
    result = tl.gather(value, index, 1)
    tl.store(y + m[:, None] * 32 + n[None, :], result)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mask", [1, 4])
def test_proved_physical_xor_is_canonicalized(enabled, mask):
    k = _compile(enabled, mask, False)
    assert bool(k.metadata.canonicalized_lane_shuffle_count) == enabled
    ir = k.asm["llir"]
    assert ("call i32 @llvm.musa.shfl.xor.sync.i32" in ir) == enabled
    assert ("call i32 @llvm.musa.shfl.idx.sync.i32" in ir) != enabled


@pytest.mark.parametrize("enabled", [False, True])
def test_dynamic_index_is_preserved(enabled):
    k = _compile(enabled, 1, True)
    assert k.metadata.canonicalized_lane_shuffle_count == 0
    assert "call i32 @llvm.musa.shfl.idx.sync.i32" in k.asm["llir"]


def _compile(enabled, mask, dynamic):
    return compile_musa(
        _lane_gather,
        {"x": "*fp32", "y": "*fp32", "delta": "i32", "L": "constexpr", "MASK": "constexpr", "DYNAMIC": "constexpr"},
        {"L": tle.gpu.BlockEncoding([1, 1], [1, 32], [4, 1], [1, 0]), "MASK": mask, "DYNAMIC": dynamic},
        options={"num_warps": 4, "canonicalize_lane_shuffles": enabled},
    )


def test_option_is_opt_in_and_part_of_cache_identity():
    _, backend = mthreads_backend()
    assert not backend.parse_options({}).canonicalize_lane_shuffles
    assert backend.parse_options({}).hash() != backend.parse_options({"canonicalize_lane_shuffles": True}).hash()
    with pytest.raises(ValueError, match="canonicalize_lane_shuffles must be a bool"):
        backend.parse_options({"canonicalize_lane_shuffles": 1})
