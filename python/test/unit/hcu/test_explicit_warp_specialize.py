"""Explicit warp domains retain CTA scope and distinct rendezvous phases."""

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu") or not hasattr(libtriton.hcu, "create_cta_barrier"):
    pytest.skip("explicit HCU warp domains required", allow_module_level=True)
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
def owner_layout():
    return gl.SharedLinearLayout([[1 << bit] for bit in range(10)], alignment=2048)


@g.constexpr_function
def matrix_layout():
    return gl.SharedLinearLayout(
        [[0, 1 << i] for i in range(5)] + [[1 << i, 0] for i in range(5)], alignment=16
    )


@g.jit
def produce(A, owner, SLOT: gl.constexpr, BAD_CONTROL: gl.constexpr):
    dest = owner.index(SLOT)._reinterpret(A.dtype.element_ty, [32, 32], matrix_layout())
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
def consume(owner, O):
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
def probe(
    A,
    B,
    O,
    FIRST_WORKER: gl.constexpr = 1,
    BAD_OUTSIDE: gl.constexpr = False,
    BAD_CONTROL: gl.constexpr = False,
):
    owner = gl.allocate_shared_memory(A.dtype.element_ty, [4, 1024], owner_layout())
    gl.warp_specialize(
        [
            (consume, (owner, O)),
            (produce, (A, owner, 1, BAD_CONTROL)),
            (produce, (B, owner, 2, False)),
        ],
        [FIRST_WORKER, 1],
        [32, 32],
    )
    owner._keep_alive()
    if BAD_OUTSIDE:
        gl.store(O, gl.cast(1, O.dtype.element_ty))


def compile_probe(base, worker, alignment=16, outside=False, divergent=False):
    source = GluonASTSource(
        probe,
        dict(
            A="*bf16",
            B="*bf16",
            O="*bf16",
            FIRST_WORKER="constexpr",
            BAD_OUTSIDE="constexpr",
            BAD_CONTROL="constexpr",
        ),
        dict(FIRST_WORKER=worker, BAD_OUTSIDE=outside, BAD_CONTROL=divergent),
        attrs={(i,): [("tt.divisibility", alignment)] for i in range(3)},
    )
    return triton.compile(
        source,
        target=GPUTarget("hip", "gfx936", 64),
        options=dict(num_warps=base, num_stages=1, terminal_readonly_publication=False),
    )


@pytest.mark.parametrize("base,worker,total", [(1, 1, 3), (1, 2, 4), (2, 1, 4)])
def test_explicit_domain_and_two_cta_phases(base, worker, total):
    kernel = compile_probe(base, worker)
    assert kernel.metadata.num_warps == total
    assert kernel.metadata.shared == 8192
    # Consumer's adjacent barriers match producer phases separated by VM0.
    # Folding only the consumer pair creates a race or unmatched rendezvous.
    # Identical producer continuations may share their two machine-code sites.
    assert kernel.asm["amdgcn"].count("\ts_barrier") in (5, 7)
    assert kernel.asm["ttgir"].count("hcu.cta_scope") == 6
    assert "s_ebarrier_sync" not in kernel.asm["amdgcn"]
    assert "hcu.raw_window" not in kernel.asm["ttgir"]


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
        compile_probe(**kwargs)


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
    a = torch.empty((1024,), device="cuda", dtype=dtype)
    b = torch.empty_like(a)
    storage = torch.empty((1536,), device="cuda", dtype=dtype)
    out = storage[256:-256]
    options = dict(num_warps=base, num_stages=1, terminal_readonly_publication=False)
    kernel = probe[(1,)](a, b, out, worker, False, False, **options)
    assert kernel.metadata.num_warps == total
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probe[(1,)](a, b, out, worker, False, False, **options)
    index = torch.arange(1024, device="cuda", dtype=torch.int32)
    for seed in range(3):
        a.view(torch.uint16).copy_(((index * 127 + seed * 13) & 65535).to(torch.uint16))
        b.view(torch.uint16).copy_(
            ((index * 33 + seed * 19 + 123) & 65535).to(torch.uint16)
        )
        before_a, before_b = a.view(torch.int16).clone(), b.view(torch.int16).clone()
        expected = (before_a.to(torch.int32) & 65535) ^ (
            before_b.to(torch.int32) & 65535
        )
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


from triton.experimental.gluon.language.hcu import wait_local_memory


@g.jit
def _local_wait_probe(A, O):

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
        _local_wait_probe,
        dict(A="*" + dtype, O="*" + dtype),
        {},
        attrs={(0,): [("tt.divisibility", 16)], (1,): [("tt.divisibility", 16)]},
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
        _local_wait_probe, dict(A="*" + dtype, O="*" + dtype), {},
        attrs={(0,): [("tt.divisibility", 16)],
               (1,): [("tt.divisibility", 16)]},
    )
    with pytest.raises(BeforeDeviceLink):
        triton.compile(source, target=GPUTarget("hip", "gfx946", 64),
                       options=dict(num_warps=4, num_stages=1))
    assert len(assembly) == 1 and "@llvm.amdgcn.s.waitcnt(i32 49279)" in assembly[0]
