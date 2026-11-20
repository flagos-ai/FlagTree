"""gfx936 ALT reads retain the affine subview when folding row offsets."""
import re

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@g.constexpr_function
def fragment_layout(waves, transpose):
    registers = [[1, 0], [2, 0], [4, 0], [0, 1], [0, 32], [0, 64]]
    lanes = [[0, 2], [0, 4], [0, 8], [0, 16], [8, 0], [16, 0]]
    if transpose:
        registers = [basis[::-1] for basis in registers]
        lanes = [basis[::-1] for basis in lanes]
    return gl.DistributedLinearLayout(registers, lanes, [[0, 0]] * (waves.bit_length() - 1), [],
                                      [128, 32] if transpose else [32, 128])


@g.jit
def _read_slice(A, Out, START: gl.constexpr, TRANSPOSE: gl.constexpr = False, WAVES: gl.constexpr = 4):
    blocked: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [WAVES, 1], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(32, 1, 4, [1, 0])
    fragment: gl.constexpr = fragment_layout(WAVES, TRANSPOSE)
    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, blocked))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, blocked))
    owner = gl.allocate_shared_memory(A.dtype.element_ty, [64, 128], shared,
                                      gl.load(A + rows[:, None] * 128 + cols[None, :]))
    window = owner.slice(START, 32)
    if TRANSPOSE:
        window = window.permute((1, 0))
    values = window.load(fragment)
    rows = gl.arange(0, 128 if TRANSPOSE else 32, layout=gl.SliceLayout(1, fragment))
    cols = gl.arange(0, 32 if TRANSPOSE else 128, layout=gl.SliceLayout(0, fragment))
    if TRANSPOSE:
        gl.store(Out + cols[None, :] * 128 + rows[:, None], values)
    else:
        gl.store(Out + rows[:, None] * 128 + cols[None, :], values)


@pytest.mark.parametrize("start", [0, 32])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("waves", [1, 2, 4, 8])
def test_alt_matrix_read_reuses_base_for_aligned_subslice(start, transpose, waves):
    source = GluonASTSource(
        _read_slice,
        {"A": "*bf16", "Out": "*bf16", "START": "constexpr", "TRANSPOSE": "constexpr", "WAVES": "constexpr"},
        {"START": start, "TRANSPOSE": transpose, "WAVES": waves},
        attrs={(i, ): [("tt.divisibility", 16)]
               for i in range(2)},
    )
    kernel = triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options={"num_warps": waves})
    reads = re.findall(
        r"^\s*ds_read_m32x16_b16_alt\s+v\[\d+:\d+\],\s+(v\d+)(?:\s+offset:(\d+))?",
        kernel.asm["amdgcn"],
        re.M,
    )
    by_base = {}
    for base, offset in reads:
        by_base.setdefault(base, set()).add(int(offset or 0))
    assert len(reads) == 8
    assert len(by_base) == 4 and all(offsets == {0, 1024} for offsets in by_base.values())


@pytest.mark.parametrize("start", [0, 32])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("waves", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_transposed_physical_alt_map_changed_replay(start, transpose, waves, dtype_name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a Hygon device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("requires gfx936")
    source = torch.empty((64, 128), dtype=getattr(torch, dtype_name), device="cuda")
    storage = torch.full((32 * 128 + 256, ), 7, dtype=source.dtype, device=source.device)
    output = storage[128:-128].view(32, 128)
    _read_slice[(1, )](source, output, start, transpose, waves, num_warps=waves)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _read_slice[(1, )](source, output, start, transpose, waves, num_warps=waves)
    for _ in range(3):
        source.copy_(torch.randn_like(source))
        graph.replay()
        torch.testing.assert_close(output, source[start:start + 32], atol=0, rtol=0)
        assert torch.all(storage[:128] == 7) and torch.all(storage[-128:] == 7)


@pytest.mark.parametrize("start", [0, 32])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("waves", [1, 2, 4, 8])
def test_alt_transpose_isomorphism_on_device(start, transpose, waves):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an HCU device")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx936":
        pytest.skip("requires gfx936")
    source = torch.randn((64, 128), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((32, 128), dtype=source.dtype, device=source.device)
    _read_slice[(1, )](source, output, start, transpose, waves, num_warps=waves)
    torch.testing.assert_close(output, source[start:start + 32], rtol=0, atol=0)
