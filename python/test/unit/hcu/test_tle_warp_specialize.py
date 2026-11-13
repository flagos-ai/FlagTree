"""Ordinary TL warp partitions use local layouts and isolated LDS indexes."""
import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import triton.language as tl
from triton.language.core import builtin
import triton.experimental.tle.language as tle

if not hasattr(libtriton, 'hcu'):
    pytest.skip('HCU compiler required', allow_module_level=True)

@builtin
def cta_phase(_semantic=None):
    # Existing typed HCU rendezvous; no inline assembly. Prototype only.
    from triton._C.libtriton import hcu
    builder = _semantic.builder
    hcu.create_cta_barrier(builder.get_op_builder(), builder.get_loc())


@triton.jit
def _copy_default(A, C):
    i = tl.arange(0, 256)
    tl.store(C + i, tl.load(A + i) + 1)


@triton.jit
def _copy_worker(A, D):
    i = tl.arange(0, 256)
    tl.store(D + i, tl.load(A + i) + 2)


@triton.jit
def role_copy(A, C, D):
    tle.gpu.warp_specialize(
        [(_copy_default, (A, C)), (_copy_worker, (A, D))], [4], [32]
    )


@triton.jit
def _shared_producer(A, owner):
    i = tl.arange(0, 256)
    owner.slot(0).store(tl.load(A + i))
    cta_phase()
    cta_phase()


@triton.jit
def _shared_consumer(C, owner):
    i = tl.arange(0, 256)
    cta_phase()
    value = owner.slot(0).load()
    tl.store(C + i, value + 3)
    cta_phase()


@triton.jit
def role_shared_round(A, C):
    # Existing HCU-compatible default chart for this interface probe only.
    # Production GEMM still needs backend-selected operand/shared layouts.
    owner = tle.gpu.alloc([1, 256], A.dtype.element_ty,
                          nv_mma_shared_layout=False)
    tle.gpu.warp_specialize(
        [(_shared_consumer, (C, owner)), (_shared_producer, (A, owner))],
        [4], [32],
    )


@triton.jit
def role_copy_configurable(A, C, D, WORKER_WARPS: tl.constexpr):
    tle.gpu.warp_specialize(
        [(_copy_default, (A, C)), (_copy_worker, (A, D))],
        [WORKER_WARPS], [32],
    )


@triton.jit
def role_three(A, C, D, E):
    tle.gpu.warp_specialize(
        [(_copy_default, (A, C)), (_copy_worker, (A, D)),
         (_copy_worker, (A, E))], [4, 4], [32, 32],
    )


from triton.language.extra.hip import accumulator_fence


@triton.jit
def _dot_worker(A, B, C):
    i = tl.arange(0, 32)
    av = tl.load(A + i[:, None] * 32 + i[None, :])
    bv = tl.load(B + i[:, None] * 32 + i[None, :])
    acc = tl.zeros((32, 32), tl.float32)
    acc = accumulator_fence(acc, 1)
    acc = tl.dot(av, bv, acc, allow_tf32=False)
    acc = accumulator_fence(acc, 0)
    tl.store(C + i[:, None] * 32 + i[None, :], acc.to(C.dtype.element_ty))


@triton.jit
def role_worker_dot(A, B, C, D):
    tle.gpu.warp_specialize(
        [(_copy_default, (A, D)), (_dot_worker, (A, B, C))], [4], [128]
    )


def _compile_role(fn, warps, constants=None, dtype="fp32", attrs=None):
    constants = constants or {}
    signature = {name: "constexpr" if name in constants else "*" + dtype
                 for name in fn.arg_names}
    return triton.compile(
        ASTSource(fn, signature, constants, attrs=attrs),
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=warps, num_stages=1, kpack=8,
                     mmac_transposed=True),
    )


@pytest.mark.parametrize("main,worker", [(8, 4), (4, 8), (4, 4)])
def test_partition_local_warp_layout(main, worker):
    kernel = _compile_role(role_copy_configurable, main,
                           dict(WORKER_WARPS=worker))
    assert kernel.metadata.num_warps >= main + worker
    assert kernel.metadata.shared == 0


def test_direct_lds_indexes_stay_in_isolated_worker():
    kernel = _compile_role(role_shared_round, 8)
    assert "hcug.buffer_load_to_local" in kernel.asm["ttgir"]
    assert kernel.metadata.num_warps == 12
    assert kernel.metadata.shared == 1024


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_worker_dot_and_priority_use_worker_layout(dtype):
    kernel = _compile_role(role_worker_dot, 8, dtype=dtype)
    assert kernel.metadata.num_warps == 12
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]


@triton.jit
def _register_worker(values, D):
    tl.store(D + tl.arange(0, 256), values)


@triton.jit
def _unencoded_register_capture(A, C, D):
    values = tl.load(A + tl.arange(0, 256))
    tle.gpu.warp_specialize(
        [(_copy_default, (A, C)), (_register_worker, (values, D))], [4], [32]
    )


def test_unencoded_register_capture_is_rejected(capfd):
    with pytest.raises(RuntimeError):
        _compile_role(_unencoded_register_capture, 8)
    assert "explicit layout transfer" in capfd.readouterr().err


@triton.jit
def _range_worker(A, D):
    i = tl.program_id(0) * 8192 + tl.arange(0, 256)
    tl.store(D + i, tl.load(A + i))


@triton.jit
def _range_roles(A, C, D):
    tle.gpu.warp_specialize(
        [(_copy_default, (A, C)), (_range_worker, (A, D))], [4], [32]
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("pointer_range", [32, 64])
def test_worker_preserves_only_proven_pointer_range(dtype, pointer_range):
    attrs = {(i,): [("tt.divisibility", 16),
                    ("tt.pointer_range", pointer_range)] for i in range(3)}
    kernel = _compile_role(_range_roles, 8, dtype=dtype, attrs=attrs)
    worker = kernel.asm["ttgir"].split("partition0", 1)[1]
    assert ("hcug.buffer_load" in worker) == (pointer_range == 32)


@triton.jit
def _carried_dot(A, B, C, K: tl.constexpr):
    m = tl.arange(0, 128)
    n = tl.arange(0, 256)
    k = tl.arange(0, 32)
    av = tl.load(A + m[:, None] * K + k[None, :])
    acc = tl.zeros((128, 256), tl.float32)
    for block in range(tl.cdiv(K, 32)):
        kk = block * 32 + k
        bv = tl.load(B + kk[:, None] * 256 + n[None, :], kk[:, None] < K, 0)
        acc = tl.dot(av, bv, acc, allow_tf32=False)
        next_k = kk + 32
        av = tl.load(A + m[:, None] * K + next_k[None, :], next_k[None, :] < K, 0)
    tl.store(C + m[:, None] * 256 + n[None, :], acc.to(C.dtype.element_ty))


@triton.jit
def role_carried_dot(A, B, C, D, K: tl.constexpr):
    tle.gpu.warp_specialize(
        [(_carried_dot, (A, B, C, K)), (_copy_default, (A, D))], [4], [32]
    )


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("k", [256, 512])
def test_prefetched_dot_operand_stays_in_register_layout(dtype, k):
    import re

    constants = dict(K=k)
    signature = {name: "constexpr" if name in constants else "*" + dtype
                 for name in role_carried_dot.arg_names}
    kernel = triton.compile(
        ASTSource(role_carried_dot, signature, constants),
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=4, num_stages=1, kpack=2,
                     mmac_warps_m=4, mmac_transposed=True),
    )
    loops = [line for line in kernel.asm["ttgir"].splitlines() if "scf.for " in line]
    assert len(loops) == 1
    assert re.search(r"#(?:ttg\.dot_op|dot)", loops[0]), loops[0]
    assert "scf.yield" in kernel.asm["ttgir"]
