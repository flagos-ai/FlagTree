"""Ordinary block-pointer loads: register descriptors or exact masked fallback."""
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@triton.jit
def descriptor_copy(A, C, origin, runtime_stride,
                    M: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                    BK: tl.constexpr, ROW: tl.constexpr, COLUMN: tl.constexpr,
                    ROW_STRIDE: tl.constexpr, COLUMN_STRIDE: tl.constexpr,
                    DYNAMIC_COLUMN: tl.constexpr):
    row = tl.program_id(0) * BM + ROW
    if ROW_STRIDE == -1:
        stride = runtime_stride
    elif ROW_STRIDE == 0:
        stride = K
    else:
        stride = ROW_STRIDE
    if DYNAMIC_COLUMN:
        column = origin
    else:
        column = tl.program_id(1) * BK + COLUMN
    pointer = tl.make_block_ptr(A, (M, K), (stride, COLUMN_STRIDE),
                                (row, column), (BM, BK), (1, 0))
    value = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    m = tl.arange(0, BM)
    n = tl.arange(0, BK)
    tl.store(C + m[:, None] * BK + n[None, :], value)


@triton.jit
def idle():
    pass


@triton.jit
def descriptor_copy_roles(A, C, origin, runtime_stride,
                          M: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
                          BK: tl.constexpr, ROW: tl.constexpr, COLUMN: tl.constexpr,
                          ROW_STRIDE: tl.constexpr, COLUMN_STRIDE: tl.constexpr,
                          DYNAMIC_COLUMN: tl.constexpr, DEFAULT_ROLE: tl.constexpr):
    if DEFAULT_ROLE:
        tle.gpu.warp_specialize([(descriptor_copy, (A, C, origin, runtime_stride, M, K, BM, BK, ROW, COLUMN, ROW_STRIDE, COLUMN_STRIDE, DYNAMIC_COLUMN)), (idle, ())], [4], [32])
    else:
        tle.gpu.warp_specialize([(idle, ()), (descriptor_copy, (A, C, origin, runtime_stride, M, K, BM, BK, ROW, COLUMN, ROW_STRIDE, COLUMN_STRIDE, DYNAMIC_COLUMN))], [4], [32])


# Legality boundaries, not a tuning table. Each excluded case exercises a
# distinct requirement of the native complete-window descriptor semantics.
CASES = [
    ("complete", 64, 64, 0, 0, 1, False, True),
    ("next_window", 128, 64, 64, 0, 1, False, True),
    ("non_power_of_two_extent", 96, 32, 32, 0, 1, False, True),
    ("partial_k", 96, 64, 64, 0, 1, False, False),
    ("unaligned_window", 64, 32, 1, 0, 1, False, False),
    ("unknown_column", 128, 64, 1, 0, 1, True, False),
    ("padded_stride", 64, 64, 0, 128, 1, False, False),
    ("non_unit_column", 64, 64, 0, 128, 2, False, False),
    ("unknown_stride", 64, 64, 0, -1, 1, False, False),
    ("empty_k", 0, 64, 0, 0, 1, False, False),
    ("descriptor_stride_limit", 8192, 64, 0, 0, 1, False, False),
]


def compile_copy(dtype, roles, case):
    name, k, bk, column, row_stride, column_stride, dynamic, expected = case
    constants = dict(M=73, K=k, BM=128, BK=bk, ROW=-3, COLUMN=column,
                     ROW_STRIDE=row_stride, COLUMN_STRIDE=column_stride,
                     DYNAMIC_COLUMN=dynamic)
    fn = descriptor_copy_roles if roles else descriptor_copy
    if roles:
        constants["DEFAULT_ROLE"] = roles != "worker"
    signature = {name: "constexpr" if name in constants else
                 "*" + dtype if name in ["A", "C"] else
                 "i64" if name == "runtime_stride" else "i32"
                 for name in fn.arg_names}
    source = ASTSource(fn, signature, constants,
                       attrs={(i,): [("tt.divisibility", 16),
                                      ("tt.pointer_range", 32)] for i in [0, 1]})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                           options=dict(prefer_wide_async_copy=True,
                                        num_warps=4, num_stages=1))


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("roles", [False, True, "worker"])
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_sync_descriptor_or_masked_fallback(tmp_path, dtype, roles, case):
    kernel = compile_copy(dtype, roles, case)
    for key, value in kernel.asm.items():
        if isinstance(value, (str, bytes)):
            (tmp_path / ("kernel." + key)).write_bytes(
                value.encode() if isinstance(value, str) else value)
    assert ("hcu.structured_register" in kernel.asm["ttgir"]) == case[-1]
    assert ("llvm.amdgcn.struct.ptr.buffer.load." in kernel.asm["llir"]) == case[-1]
    assert "hcug.matrix_load_to_local" not in kernel.asm["ttgir"]
    assert "ttg.async_copy_global_to_local" not in kernel.asm["ttgir"]
    assert "tt.matrix_load " not in kernel.asm["ttir"] or "tt.load.async = false" in kernel.asm["ttir"]
