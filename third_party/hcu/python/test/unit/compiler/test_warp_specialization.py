# explicit warp specialize
"""Explicit warp domains retain CTA scope and distinct rendezvous phases."""

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu") or not hasattr(libtriton.hcu, "create_cta_barrier"):
    pytest.skip("explicit HCU warp domains required", allow_module_level=True)

from triton.experimental.gluon.language.hcu import wait_local_memory
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.hcu import (
    matrix_load_to_shared,
    cta_barrier,
    warp_group_id,
)
from triton.experimental.gluon.language.amd.cdna4.async_copy import (
    commit_group,
    wait_group,
)


@g.constexpr_function
def _explicit_warp_specialize_owner_layout():
    return gl.SharedLinearLayout([[1 << bit] for bit in range(10)], alignment=2048)


@g.constexpr_function
def _explicit_warp_specialize_matrix_layout():
    return gl.SharedLinearLayout([[0, 1 << i] for i in range(5)] + [[1 << i, 0] for i in range(5)], alignment=16)


@g.jit
def _explicit_warp_specialize_produce(A, owner, SLOT: gl.constexpr, BAD_CONTROL: gl.constexpr):
    dest = owner.index(SLOT)._reinterpret(A.dtype.element_ty, [32, 32], _explicit_warp_specialize_matrix_layout())
    if BAD_CONTROL:
        if warp_group_id(1) == 0:
            matrix_load_to_shared(dest, A, (32, 32), (32, 1), (0, 0))
            commit_group()
    else:
        matrix_load_to_shared(dest, A, (32, 32), (32, 1), (0, 0))
        commit_group()
    cta_barrier()
    wait_group(0)
    cta_barrier()


@g.jit
def _explicit_warp_specialize_consume(owner, O):
    cta_barrier()
    cta_barrier()
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
    left = owner.index(1).load(layout).to(gl.uint16, bitcast=True)
    right = owner.index(2).load(layout).to(gl.uint16, bitcast=True)
    result = left ^ right
    i = gl.arange(0, 1024, layout=layout)
    # A wave-dependent consumer branch must not invalidate the independent
    # producer partition's uniform origins/control proof.
    if warp_group_id(1) == 0:
        gl.store(O + i, (result ^ 0x10).to(O.dtype.element_ty, bitcast=True))
    else:
        gl.store(O + i, (result ^ 0x20).to(O.dtype.element_ty, bitcast=True))


@g.jit
def _explicit_warp_specialize_probe(
    A,
    B,
    O,
    FIRST_WORKER: gl.constexpr = 1,
    BAD_OUTSIDE: gl.constexpr = False,
    BAD_CONTROL: gl.constexpr = False,
):
    owner = gl.allocate_shared_memory(A.dtype.element_ty, [4, 1024], _explicit_warp_specialize_owner_layout())
    gl.warp_specialize(
        [
            (_explicit_warp_specialize_consume, (owner, O)),
            (_explicit_warp_specialize_produce, (A, owner, 1, BAD_CONTROL)),
            (_explicit_warp_specialize_produce, (B, owner, 2, False)),
        ],
        [FIRST_WORKER, 1],
        [32, 32],
    )
    owner._keep_alive()
    if BAD_OUTSIDE:
        gl.store(O, gl.cast(1, O.dtype.element_ty))


def _explicit_warp_specialize_compile_probe(base, worker, alignment=16, outside=False, divergent=False):
    source = GluonASTSource(
        _explicit_warp_specialize_probe,
        dict(
            A="*bf16",
            B="*bf16",
            O="*bf16",
            FIRST_WORKER="constexpr",
            BAD_OUTSIDE="constexpr",
            BAD_CONTROL="constexpr",
        ),
        dict(FIRST_WORKER=worker, BAD_OUTSIDE=outside, BAD_CONTROL=divergent),
        attrs={(i, ): [("tt.divisibility", alignment)]
               for i in range(3)},
    )
    return triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=base, num_stages=1, terminal_readonly_publication=False),
    )


@pytest.mark.parametrize("base,worker,total", [(1, 1, 3), (1, 2, 4), (2, 1, 4)])
def test_explicit_domain_and_two_cta_phases(base, worker, total):
    kernel = _explicit_warp_specialize_compile_probe(base, worker)
    assert kernel.metadata.num_warps == total
    assert kernel.metadata.shared == 8192
    # Consumer's adjacent barriers match producer phases separated by VM0.
    # Folding only the consumer pair creates a race or unmatched rendezvous.
    # Identical producer continuations may share their two machine-code sites.
    assert kernel.asm["amdgcn"].count("\ts_barrier") in (5, 7)
    assert kernel.asm["ttgir"].count("hcu.cta_scope") == 6
    assert "s_ebarrier_sync" not in kernel.asm["amdgcn"]
    assert "hcu.raw_window" not in kernel.asm["ttgir"]


@pytest.mark.parametrize("base,worker", [(1, 1), (1, 2), (2, 1)])
def test_private_function_does_not_inherit_kernel_capture_abi(tmp_path, capfd, base, worker):
    text = _explicit_warp_specialize_compile_probe(base, worker).asm["ttgir"]
    assert "tt.func public" in text
    path = tmp_path / "private-warp.ttgir"
    path.write_text(text.replace("tt.func public", "tt.func private", 1))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    from triton.backends.hcu.compiler import HIPBackend
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_warp_specialize_to_llvm(manager, "gfx936", 0, 0, False, 0, 0, 0)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        manager.run(module, "private_warp_domain")
    assert "requires a void terminal kernel" in capfd.readouterr().err


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(base=1, worker=1, alignment=8),
        dict(base=1, worker=1, outside=True),
        dict(base=1, worker=2, divergent=True),
        dict(base=8, worker=8),
    ],
)
def test_unsafe_explicit_domain_rejected(kwargs):
    with pytest.raises(Exception):
        _explicit_warp_specialize_compile_probe(**kwargs)


@pytest.mark.parametrize("base,worker,total", [(1, 1, 3), (1, 2, 4), (2, 1, 4)])
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
def test_partition_captures_and_changed_input_replay(base, worker, total, dtype_name):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("HCU GPU required")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("gfx936 required")
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    a = torch.empty((1024, ), device="cuda", dtype=dtype)
    b = torch.empty_like(a)
    storage = torch.empty((1536, ), device="cuda", dtype=dtype)
    out = storage[256:-256]
    options = dict(num_warps=base, num_stages=1, terminal_readonly_publication=False)
    kernel = _explicit_warp_specialize_probe[(1, )](a, b, out, worker, False, False, **options)
    assert kernel.metadata.num_warps == total
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _explicit_warp_specialize_probe[(1, )](a, b, out, worker, False, False, **options)
    index = torch.arange(1024, device="cuda", dtype=torch.int32)
    for seed in range(3):
        a.view(torch.uint16).copy_(((index * 127 + seed * 13) & 65535).to(torch.uint16))
        b.view(torch.uint16).copy_(((index * 33 + seed * 19 + 123) & 65535).to(torch.uint16))
        before_a, before_b = a.view(torch.int16).clone(), b.view(torch.int16).clone()
        expected = (before_a.to(torch.int32) & 65535) ^ (before_b.to(torch.int32) & 65535)
        mask = torch.where(index < 512, 0x10, 0x20) if base == 2 else 0x10
        expected = (expected ^ mask).to(torch.int16)
        for _ in range(3):
            storage.view(torch.int16).fill_(0x3333)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), expected)
            assert torch.equal(a.view(torch.int16), before_a)
            assert torch.equal(b.view(torch.int16), before_b)
            assert torch.all(storage[:256].view(torch.int16) == 0x3333)
            assert torch.all(storage[-256:].view(torch.int16) == 0x3333)


@g.jit
def _explicit_warp_specialize_local_wait_probe(A, O):

    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    shared: gl.constexpr = gl.SharedLinearLayout([[1], [2], [4], [8], [16], [32], [64], [128]])
    offset = gl.arange(0, 256, layout=layout)
    owner = gl.allocate_shared_memory(A.dtype.element_ty, [256], shared)
    owner.store(gl.load(A + offset))
    gl.thread_barrier()
    value = owner.load(layout)
    wait_local_memory()
    gl.store(O + offset, value)


@pytest.mark.parametrize("arch", ["gfx936"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_local_memory_wait_uses_existing_target_counter(arch, dtype):
    source = GluonASTSource(
        _explicit_warp_specialize_local_wait_probe,
        dict(A="*" + dtype, O="*" + dtype),
        {},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("hip", arch, 64),
        options=dict(num_warps=4, num_stages=1),
    )
    assert "memory_counter_wait ds(0)" in kernel.asm["ttgir"]
    assert "@llvm.amdgcn.s.waitcnt(i32 49279)" in kernel.asm["llir"]


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_local_wait_gfx946_llvm_before_unavailable_device_codegen(monkeypatch, dtype):
    # The gfx936 validation image has no gfx946 ROCm device bitcode. Validate
    # real gfx946 LLVM lowering, and explicitly stop before native codegen.
    from triton.backends.hcu.compiler import HIPBackend

    class BeforeDeviceLink(Exception):
        pass

    assembly = []

    def capture(src, metadata, options):
        assert options.arch == "gfx946"
        assembly.append(src)
        raise BeforeDeviceLink()

    monkeypatch.setattr(HIPBackend, "make_amdgcn", staticmethod(capture))
    source = GluonASTSource(
        _explicit_warp_specialize_local_wait_probe,
        dict(A="*" + dtype, O="*" + dtype),
        {},
        attrs={(0, ): [("tt.divisibility", 16)], (1, ): [("tt.divisibility", 16)]},
    )
    with pytest.raises(BeforeDeviceLink):
        triton.compile(source, target=GPUTarget("hip", "gfx946", 64), options=dict(num_warps=4, num_stages=1))
    assert len(assembly) == 1 and "@llvm.amdgcn.s.waitcnt(i32 49279)" in assembly[0]


# tle warp specialize
from triton.compiler import ASTSource
import triton.language as tl
from triton.language.core import builtin
import triton.experimental.tle.language as tle

if not hasattr(libtriton, 'hcu'):
    pytest.skip('HCU compiler required', allow_module_level=True)

from triton.language.extra.hip import accumulator_fence


@builtin
def _tle_warp_specialize_cta_phase(_semantic=None):
    # Existing typed HCU rendezvous; no inline assembly. Prototype only.
    from triton._C.libtriton import hcu
    builder = _semantic.builder
    hcu.create_cta_barrier(builder.get_op_builder(), builder.get_loc())


@triton.jit
def _tle_warp_specialize_copy_default(A, C):
    i = tl.arange(0, 256)
    tl.store(C + i, tl.load(A + i) + 1)


@triton.jit
def _tle_warp_specialize_copy_worker(A, D):
    i = tl.arange(0, 256)
    tl.store(D + i, tl.load(A + i) + 2)


@triton.jit
def _tle_warp_specialize_role_copy(A, C, D):
    tle.gpu.warp_specialize([(_tle_warp_specialize_copy_default, (A, C)), (_tle_warp_specialize_copy_worker, (A, D))],
                            [4], [32])


@triton.jit
def _tle_warp_specialize_shared_producer(A, owner):
    i = tl.arange(0, 256)
    owner.slot(0).store(tl.load(A + i))
    _tle_warp_specialize_cta_phase()
    _tle_warp_specialize_cta_phase()


@triton.jit
def _tle_warp_specialize_shared_consumer(C, owner):
    i = tl.arange(0, 256)
    _tle_warp_specialize_cta_phase()
    value = owner.slot(0).load()
    tl.store(C + i, value + 3)
    _tle_warp_specialize_cta_phase()


@triton.jit
def _tle_warp_specialize_role_shared_round(A, C):
    # Existing HCU-compatible default chart for this interface probe only.
    # Production GEMM still needs backend-selected operand/shared layouts.
    owner = tle.gpu.alloc([1, 256], A.dtype.element_ty, nv_mma_shared_layout=False)
    tle.gpu.warp_specialize(
        [(_tle_warp_specialize_shared_consumer, (C, owner)), (_tle_warp_specialize_shared_producer, (A, owner))],
        [4],
        [32],
    )


@triton.jit
def _tle_warp_specialize_role_copy_configurable(A, C, D, WORKER_WARPS: tl.constexpr):
    tle.gpu.warp_specialize(
        [(_tle_warp_specialize_copy_default, (A, C)), (_tle_warp_specialize_copy_worker, (A, D))],
        [WORKER_WARPS],
        [32],
    )


@triton.jit
def _tle_warp_specialize_role_three(A, C, D, E):
    tle.gpu.warp_specialize(
        [(_tle_warp_specialize_copy_default, (A, C)), (_tle_warp_specialize_copy_worker, (A, D)),
         (_tle_warp_specialize_copy_worker, (A, E))],
        [4, 4],
        [32, 32],
    )


@triton.jit
def _tle_warp_specialize_dot_worker(A, B, C):
    i = tl.arange(0, 32)
    av = tl.load(A + i[:, None] * 32 + i[None, :])
    bv = tl.load(B + i[:, None] * 32 + i[None, :])
    acc = tl.zeros((32, 32), tl.float32)
    acc = accumulator_fence(acc, 1)
    acc = tl.dot(av, bv, acc, allow_tf32=False)
    acc = accumulator_fence(acc, 0)
    tl.store(C + i[:, None] * 32 + i[None, :], acc.to(C.dtype.element_ty))


@triton.jit
def _tle_warp_specialize_role_worker_dot(A, B, C, D):
    tle.gpu.warp_specialize([(_tle_warp_specialize_copy_default, (A, D)), (_tle_warp_specialize_dot_worker, (A, B, C))],
                            [4], [128])


def _tle_warp_specialize_compile_role(fn, warps, constants=None, dtype="fp32", attrs=None):
    constants = constants or {}
    signature = {name: "constexpr" if name in constants else "*" + dtype for name in fn.arg_names}
    return triton.compile(
        ASTSource(fn, signature, constants, attrs=attrs),
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=warps, num_stages=1, kpack=8, mmac_transposed=True),
    )


@pytest.mark.parametrize("main,worker", [(8, 4), (4, 8), (4, 4)])
def test_partition_local_warp_layout(main, worker):
    kernel = _tle_warp_specialize_compile_role(_tle_warp_specialize_role_copy_configurable, main,
                                               dict(WORKER_WARPS=worker))
    assert kernel.metadata.num_warps >= main + worker
    assert kernel.metadata.shared == 0


def test_direct_lds_indexes_stay_in_isolated_worker():
    kernel = _tle_warp_specialize_compile_role(_tle_warp_specialize_role_shared_round, 8)
    assert "hcug.buffer_load_to_local" in kernel.asm["ttgir"]
    assert kernel.metadata.num_warps == 12
    assert kernel.metadata.shared == 1024


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_worker_dot_and_priority_use_worker_layout(dtype):
    kernel = _tle_warp_specialize_compile_role(_tle_warp_specialize_role_worker_dot, 8, dtype=dtype)
    assert kernel.metadata.num_warps == 12
    assert kernel.asm["ttgir"].count("hcug.accumulator_fence") == 2
    assert "s_setprio 1" in kernel.asm["amdgcn"]
    assert "s_setprio 0" in kernel.asm["amdgcn"]
    assert any("<4 x float>" in line and "asm sideeffect" in line and "s_setprio" in line
               for line in kernel.asm["llir"].splitlines())


@triton.jit
def _tle_warp_specialize_register_worker(values, D):
    tl.store(D + tl.arange(0, 256), values)


@triton.jit
def _tle_warp_specialize_unencoded_register_capture(A, C, D):
    values = tl.load(A + tl.arange(0, 256))
    tle.gpu.warp_specialize([(_tle_warp_specialize_copy_default, (A, C)),
                             (_tle_warp_specialize_register_worker, (values, D))], [4], [32])


def test_unencoded_register_capture_is_rejected(capfd):
    with pytest.raises(RuntimeError):
        _tle_warp_specialize_compile_role(_tle_warp_specialize_unencoded_register_capture, 8)
    assert "explicit layout transfer" in capfd.readouterr().err


@triton.jit
def _tle_warp_specialize_range_worker(A, D):
    i = tl.program_id(0) * 8192 + tl.arange(0, 256)
    tl.store(D + i, tl.load(A + i))


@triton.jit
def _tle_warp_specialize_range_roles(A, C, D):
    tle.gpu.warp_specialize([(_tle_warp_specialize_copy_default, (A, C)), (_tle_warp_specialize_range_worker, (A, D))],
                            [4], [32])


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("pointer_range", [32, 64])
def test_worker_preserves_only_proven_pointer_range(dtype, pointer_range):
    attrs = {(i, ): [("tt.divisibility", 16), ("tt.pointer_range", pointer_range)] for i in range(3)}
    kernel = _tle_warp_specialize_compile_role(_tle_warp_specialize_range_roles, 8, dtype=dtype, attrs=attrs)
    worker = kernel.asm["ttgir"].split("partition0", 1)[1]
    assert ("hcug.buffer_load" in worker) == (pointer_range == 32)


@triton.jit
def _tle_warp_specialize_carried_dot(A, B, C, K: tl.constexpr):
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
def _tle_warp_specialize_role_carried_dot(A, B, C, D, K: tl.constexpr):
    tle.gpu.warp_specialize([(_tle_warp_specialize_carried_dot, (A, B, C, K)),
                             (_tle_warp_specialize_copy_default, (A, D))], [4], [32])


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("k", [256, 512])
def test_prefetched_dot_operand_stays_in_register_layout(dtype, k):
    import re

    constants = dict(K=k)
    signature = {
        name: "constexpr" if name in constants else "*" + dtype
        for name in _tle_warp_specialize_role_carried_dot.arg_names
    }
    kernel = triton.compile(
        ASTSource(_tle_warp_specialize_role_carried_dot, signature, constants),
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=4, num_stages=1, kpack=2, mmac_warps_m=4, mmac_transposed=True),
    )
    loops = [line for line in kernel.asm["ttgir"].splitlines() if "scf.for " in line]
    assert len(loops) == 1
    assert re.search(r"#(?:ttg\.dot_op|dot)", loops[0]), loops[0]
    assert "scf.yield" in kernel.asm["ttgir"]


# tle structured captures

from triton.backends.hcu.compiler import HIPBackend


def _tle_structured_captures_lower(tmp_path, dtype, alignment, divergent=False, control=None):
    extra = ('%offset = "hcug.warp_group_id"() {groupSize = 1 : i32} : () -> i32\n'
             "    %derived = tt.addptr %p, %offset : !tt.ptr<DT>, i32").replace("DT", dtype) if divergent else ""
    base = "%derived" if divergent else "%p"
    text = f'''#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true, mmacLayout = 1}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @captured_descriptor(%p: !tt.ptr<{dtype}> {{tt.divisibility = {alignment} : i32}}) {{
    {extra}
    ttg.warp_specialize({base}) attributes {{requestedRegisters = array<i32: 32>}}
    default {{ ttg.warp_yield }}
    partition0(%base: !tt.ptr<{dtype}>) num_warps(4) {{
      %zero = arith.constant 0 : i32
      %one = arith.constant 1 : i32
      %M = arith.constant 128 : i32
      %K = arith.constant 64 : i32
      %stride = arith.constant 64 : i64
      %unit_stride = arith.constant 1 : i64
      %tile = tt.matrix_load %base, shape = [%M, %K], strides = [%stride, %unit_stride],
        tensorShape = [128, 32], indices = [%zero, %zero, %one]
        {{hcu.structured_register}} : <{dtype}> -> tensor<128x32x{dtype}, #ttg.dot_op<{{opIdx = 0, parent = #mma, kWidth = 8}}>>
      ttg.warp_return
    }} : (!tt.ptr<{dtype}>) -> ()
    tt.return
  }}
}}'''
    if control is not None:
        # Isolate structured-register execution from warp-specialize structure.
        begin = text.index("    ttg.warp_specialize(")
        body = text.index("      %zero =", begin)
        end = text.index("      ttg.warp_return", body)
        load = text[body:end].replace("%base", "%p")
        selector = ('%tid = "hcug.warp_group_id"() {groupSize = 1 : i32} : () -> i32'
                    if control == "wave" else "%tid = rocdl.workitem.id.x : i32")
        prefix = ("    " + selector + "\n    %limit = arith.constant 1 : i32\n"
                  "    %take = arith.cmpi ult, %tid, %limit : i32\n")
        if control == "reconverged":
            code = ("    cf.cond_br %take, ^left, ^right\n  ^left:\n    cf.br ^join\n"
                    "  ^right:\n    cf.br ^join\n  ^join:\n" + load + "    tt.return\n")
        else:
            code = ("    cf.cond_br %take, ^load, ^exit\n  ^load:\n" + load +
                    "    cf.br ^exit\n  ^exit:\n    tt.return\n")
        text = text[:begin] + prefix + code + "  }\n}"
    path = tmp_path / "capture.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.passes.ttgpuir.add_allocate_warp_groups(manager)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "structured_capture")
    return str(module)


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("alignment,width", [(16, "v4i32"), (2, "i16")])
def test_descriptor_capture_uses_abi_base_and_alignment(tmp_path, dtype, alignment, width):
    result = _tle_structured_captures_lower(tmp_path, dtype, alignment)
    assert "llvm.amdgcn.struct.ptr.buffer.load." + width in result


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("alignment", [2, 16])
def test_capture_does_not_make_wave_varying_pointer_uniform(tmp_path, capfd, dtype, alignment):
    with pytest.raises(RuntimeError):
        _tle_structured_captures_lower(tmp_path, dtype, alignment, True)
    diagnostic = capfd.readouterr().err
    (tmp_path / "diagnostic.log").write_text(diagnostic)
    assert "uniform AS1 base" in diagnostic


@pytest.mark.parametrize("control,accepted", [("reconverged", True), ("wave", True), ("lane", False)])
def test_structured_register_control_is_site_relative(tmp_path, capfd, control, accepted):
    if accepted:
        result = _tle_structured_captures_lower(tmp_path, "bf16", 16, control=control)
        assert "llvm.amdgcn.struct.ptr.buffer.load." in result
    else:
        with pytest.raises(RuntimeError):
            _tle_structured_captures_lower(tmp_path, "bf16", 16, control=control)
        assert "execution control" in capfd.readouterr().err


# tle uniform division


def _tle_uniform_division_lower(tmp_path, divisor, varying=False, dynamic=False):
    leaf = ("%source = rocdl.workitem.id.x : i32" if varying else "%source = tt.get_program_id x : i32")
    divisor_value = "%dynamic" if dynamic else "%divisor"
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}}>
#shared = #ttg.swizzled_shared<{{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}}>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"}} {{
  tt.func public @copy(%p: !tt.ptr<bf16> {{tt.divisibility = 16 : i32,
      tt.pointer_range = 32 : i32}}, %dynamic: i32) {{
    %divisor = arith.constant {divisor} : i32
    %width = arith.constant 256 : i32
    {leaf}
    %group = arith.remui %source, {divisor_value} : i32
    %origin = arith.muli %group, %width : i32
    %columns = tt.make_range {{end = 256 : i32, start = 0 : i32}}
      : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %base = tt.splat %origin
      : i32 -> tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %column = arith.addi %base, %columns
      : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
    %expanded = tt.expand_dims %column {{axis = 0 : i32}}
      : tensor<256xi32, #ttg.slice<{{dim = 0, parent = #blocked}}>>
      -> tensor<1x256xi32, #blocked>
    %stride = arith.constant dense<2048> : tensor<1x256xi32, #blocked>
    %rows = arith.muli %expanded, %stride : tensor<1x256xi32, #blocked>
    %row_offsets = tt.broadcast %rows
      : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %krange = tt.make_range {{end = 64 : i32, start = 0 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
    %kexpanded = tt.expand_dims %krange {{axis = 1 : i32}}
      : tensor<64xi32, #ttg.slice<{{dim = 1, parent = #blocked}}>>
      -> tensor<64x1xi32, #blocked>
    %k = tt.broadcast %kexpanded
      : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %offset = arith.addi %row_offsets, %k : tensor<64x256xi32, #blocked>
    %dest = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #shared, #smem, mutable>
    %copy = hcug.buffer_load_to_local %p[%offset] into %dest
      {{contiguity = 8 : i32}} : <bf16>[tensor<64x256xi32, #blocked>]
      -> <64x256xbf16, #shared, #smem, mutable>
    %commit = ttg.async_commit_group tokens %copy
    %wait = hcug.async_wait %commit {{num_inst = 0 : i32}}
    tt.return
  }}
}}'''
    path = tmp_path / "uniform_division.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    manager = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_allocate_shared_memory(manager)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(manager, "gfx936", False)
    manager.run(module, "uniform_division")
    return str(module)


@pytest.mark.parametrize("divisor", [3, 4, 5, 12, 192])
def test_uniform_nonzero_constant_division_needs_no_lane_shuffle(tmp_path, divisor):
    assert "bpermute" not in _tle_uniform_division_lower(tmp_path, divisor)


@pytest.mark.parametrize("divisor,varying,dynamic", [
    (12, True, False),
    (192, True, False),
    (12, False, True),
])
def test_lane_varying_and_unproved_divisors_keep_original_shuffle(tmp_path, divisor, varying, dynamic):
    assert "bpermute" in _tle_uniform_division_lower(tmp_path, divisor, varying, dynamic)


# memory boundaries


@g.jit
def _memory_boundaries_partition(O, VALUE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = gl.arange(0, 128, layout=layout)
    # This rendezvous belongs to this partition, not the entire physical CTA.
    gl.thread_barrier()
    gl.store(O + i + VALUE * 128, i + VALUE)


@g.jit
def _memory_boundaries_local_barrier(O):
    gl.warp_specialize([(_memory_boundaries_partition, (O, 0)), (_memory_boundaries_partition, (O, 1))], [1], [32])


@g.jit
def _memory_boundaries_multiwave_local_barrier(O, WORKER: gl.constexpr):
    gl.warp_specialize([(_memory_boundaries_partition, (O, 0)), (_memory_boundaries_partition, (O, 1))], [WORKER], [32])


def _memory_boundaries_compile(fn, constants, warps):
    signature = dict(O="*i32", **{k: "constexpr" for k in constants})
    return triton.compile(GluonASTSource(fn, signature, constants), target=GPUTarget("hip", "gfx936", 64),
                          options=dict(num_warps=warps, num_stages=1, terminal_readonly_publication=False))


def test_single_wave_partition_barrier_uses_wave_ordering():
    kernel = _memory_boundaries_compile(_memory_boundaries_local_barrier, {}, 1)
    assert kernel.metadata.num_warps == 2
    assert "llvm.amdgcn.wave.barrier" in kernel.asm["llir"]
    assert "s_ebarrier_sync" not in kernel.asm["amdgcn"]


@pytest.mark.parametrize("base,worker", [(1, 2), (2, 1), (2, 2)])
def test_multiwave_partition_barrier_has_explicit_diagnostic(base, worker, capfd):
    with pytest.raises(Exception):
        _memory_boundaries_compile(_memory_boundaries_multiwave_local_barrier, dict(WORKER=worker), base)
    assert "partition-local barriers require a single wave" in capfd.readouterr().err


@g.jit
def _memory_boundaries_void_asm(O):
    gl.inline_asm_elementwise("s_nop 0", "", [], dtype=(), is_pure=False, pack=1)
    gl.store(O, 1)


def test_unsupported_resultless_asm_diagnosed_before_result_index(capfd):
    with pytest.raises(Exception):
        _memory_boundaries_compile(_memory_boundaries_void_asm, {}, 1)
    assert "HCU elementwise inline asm requires at least one result" in capfd.readouterr().err


@g.jit
def _memory_boundaries_ordinary_asm(O):
    value = gl.inline_asm_elementwise("v_mov_b32 $0, $1", "=v,v", [7], dtype=gl.int32, is_pure=False, pack=1)
    gl.store(O, value)


def test_result_asm_still_compiles():
    kernel = _memory_boundaries_compile(_memory_boundaries_ordinary_asm, {}, 1)
    assert "v_mov_b32" in kernel.asm["amdgcn"]


def test_single_wave_partition_barrier_changed_replay():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("HCU GPU required")
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != "gfx936":
        pytest.skip("gfx936 GPU required")
    storage = torch.full((512, ), -17, device="cuda", dtype=torch.int32)
    out = storage[128:384]
    _memory_boundaries_local_barrier[(1, )](out, num_warps=1, num_stages=1, terminal_readonly_publication=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _memory_boundaries_local_barrier[(1, )](out, num_warps=1, num_stages=1, terminal_readonly_publication=False)
    expected = torch.cat([torch.arange(128, device="cuda"), torch.arange(128, device="cuda") + 1]).int()
    for sentinel in [-1, -7, -123]:
        out.fill_(sentinel)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        assert torch.all(storage[:128] == -17) and torch.all(storage[384:] == -17)


@g.jit
def _memory_boundaries_partition_shared(A, O, owner, INDEX: gl.constexpr):
    write_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    read_layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    write_i = gl.arange(0, 128, layout=write_layout)
    buf = owner.index(INDEX)
    buf.store(gl.load(A + INDEX * 128 + write_i))
    gl.thread_barrier()
    # Different lane ownership forces real inter-lane LDS communication.
    values = buf.load(read_layout)
    read_i = gl.arange(0, 128, layout=read_layout)
    gl.store(O + INDEX * 128 + read_i, values)


@g.jit
def _memory_boundaries_shared_local_barrier(A, O):
    layout: gl.constexpr = gl.SharedLinearLayout([[1], [2], [4], [8], [16], [32], [64]], alignment=512)
    owner = gl.allocate_shared_memory(gl.int32, [2, 128], layout)
    gl.warp_specialize([(_memory_boundaries_partition_shared, (A, O, owner, 0)),
                        (_memory_boundaries_partition_shared, (A, O, owner, 1))], [1], [32])
    owner._keep_alive()


def test_single_wave_partition_actual_lds_changed_replay():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("HCU GPU required")
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != "gfx936":
        pytest.skip("gfx936 GPU required")
    a = torch.empty((256, ), device="cuda", dtype=torch.int32)
    storage = torch.full((512, ), -17, device="cuda", dtype=torch.int32)
    out = storage[128:384]
    options = dict(num_warps=1, num_stages=1, terminal_readonly_publication=False)
    kernel = _memory_boundaries_shared_local_barrier[(1, )](a, out, **options)
    assert "llvm.amdgcn.wave.barrier" in kernel.asm["llir"]
    assert "ds_write" in kernel.asm["amdgcn"] and "ds_read" in kernel.asm["amdgcn"]
    assert "s_ebarrier_sync" not in kernel.asm["amdgcn"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _memory_boundaries_shared_local_barrier[(1, )](a, out, **options)
    for seed in [3, 17, 101]:
        a.copy_(torch.arange(256, device="cuda", dtype=torch.int32) * seed - 71)
        before = a.clone()
        out.fill_(-123)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, before) and torch.equal(a, before)
        assert torch.all(storage[:128] == -17) and torch.all(storage[384:] == -17)


# ws allocation witness

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)

_ws_allocation_witness_compiler = pytest.importorskip("triton.backends.hcu.compiler")


def _ws_allocation_witness_allocation_size(tmp_path, extra_attributes, terminal=True):
    source = tmp_path / "warp-scratch.mlir"
    extra = ", " + extra_attributes if extra_attributes else ""
    source.write_text('''module attributes {"ttg.num-warps" = 4 : i32,
      "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32,
      ttg.target = "hip:gfx936"EXTRA} {
      tt.func public @capture(%input: i32, %out: !tt.ptr<i32>) {
        ttg.warp_specialize(%input)
        default {
          ttg.warp_yield
        }
        partition0(%captured: i32) num_warps(4) {
          ttg.warp_return
        } : (i32) -> ()
        EXIT
        tt.return
      }
    }
    '''.replace("EXTRA", extra).replace("EXIT", "" if terminal else "tt.store %out, %input : !tt.ptr<i32>"))
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    _ws_allocation_witness_compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(source), context)
    pm = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_allocate_shared_memory(pm)
    pm.run(module, "generic_ws_scratch_without_issued_raw_proof")
    return module.get_int_attr("ttg.shared")


@pytest.mark.parametrize("attributes", ["ttg.hcu.raw_window_ws = 1 : i32, ttg.hcu.raw_window_ws.internal_witness"])
@pytest.mark.parametrize("terminal", [True, False])
def test_unissued_markers_do_not_override_actual_domain(tmp_path, attributes, terminal):
    normal = _ws_allocation_witness_allocation_size(tmp_path, "", terminal)
    # Actual terminal uniform-capture IR uses direct partition branches and
    # needs no scratch. A store after WS defeats that proof and must retain it.
    assert normal == 0 if terminal else normal > 0
    assert _ws_allocation_witness_allocation_size(tmp_path, attributes, terminal) == normal


# target memory contract

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU bindings required", allow_module_level=True)
from triton.experimental.gluon.language.hcu import warp_group_id


def _target_memory_contract_parse(text, tmp_path):
    path = tmp_path / "target.mlir"
    path.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    return libtriton.ir.parse_mlir_module(str(path), context), context


@pytest.mark.parametrize("name", ["raw_window_copy", "raw_window_read", "raw_window_wait"])
def test_retired_raw_ops_are_unregistered(name, tmp_path, capfd):
    # A quoted generic syntax avoids relying on a retired custom parser.
    with pytest.raises(RuntimeError):
        _target_memory_contract_parse('module { "hcug.' + name + '"() : () -> () }', tmp_path)
    assert "unregistered operation" in capfd.readouterr().err


def _target_memory_contract_mls_argument(version):
    return '''#layout = #ttg.mls_shared<{opIdx = 0, mlsTile = [64, 32],
      elemBitWidth = 16, alt2Kind = 0, version = VERSION, order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32,
    "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @test(%tile: !ttg.memdesc<64x32xbf16, #layout,
      #ttg.shared_memory, mutable>) { tt.return }
}'''.replace("VERSION", str(version))


@pytest.mark.parametrize("version", [0, 256, 257, 258, 259])
def test_retired_physical_versions_fail_before_vendor_lowering(version, tmp_path, capfd):
    module, context = _target_memory_contract_parse(_target_memory_contract_mls_argument(version), tmp_path)
    pm = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx936", False)
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        pm.run(module, "retired_physical_encoding")
    assert "retired custom HCU MLS physical encoding" in capfd.readouterr().err


def test_vendor_mls_layout_parser_remains_available(tmp_path):
    module, _ = _target_memory_contract_parse(_target_memory_contract_mls_argument(1), tmp_path)
    assert "version = 1" in str(module)


@g.jit
def _target_memory_contract_group_query(O, SIZE: gl.constexpr):
    gl.store(O, warp_group_id(SIZE))


def _target_memory_contract_compile_query(size):
    source = GluonASTSource(_target_memory_contract_group_query, {"O": "*i32", "SIZE": "constexpr"}, {"SIZE": size})
    return triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options=dict(num_warps=4, num_stages=1))


@pytest.mark.parametrize("size", [1, 2, 4])
def test_partition_relative_query_without_raw_api(size):
    kernel = _target_memory_contract_compile_query(size)
    assert "hcug.warp_group_id" in kernel.asm["ttgir"]
    assert "raw_window" not in kernel.asm["ttgir"]


@pytest.mark.parametrize("size", [8, 16])
def test_query_cannot_exceed_actual_partition(size, capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _target_memory_contract_compile_query(size)
    assert "within this partition" in capfd.readouterr().err
