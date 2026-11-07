"""Actual gfx936 lowering of partition rendezvous and unsupported void asm."""
import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)


@g.jit
def _partition(O, VALUE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    i = gl.arange(0, 128, layout=layout)
    # This rendezvous belongs to this partition, not the entire physical CTA.
    gl.thread_barrier()
    gl.store(O + i + VALUE * 128, i + VALUE)


@g.jit
def _local_barrier(O):
    gl.warp_specialize([(_partition, (O, 0)), (_partition, (O, 1))], [1], [32])


@g.jit
def _multiwave_local_barrier(O, WORKER: gl.constexpr):
    gl.warp_specialize([(_partition, (O, 0)), (_partition, (O, 1))], [WORKER], [32])


def _compile(fn, constants, warps):
    signature = dict(O="*i32", **{k: "constexpr" for k in constants})
    return triton.compile(GluonASTSource(fn, signature, constants),
                          target=GPUTarget("hip", "gfx936", 64),
                          options=dict(num_warps=warps, num_stages=1,
                                       terminal_readonly_publication=False))


def test_single_wave_partition_barrier_uses_wave_ordering():
    kernel = _compile(_local_barrier, {}, 1)
    assert kernel.metadata.num_warps == 2
    assert "llvm.amdgcn.wave.barrier" in kernel.asm["llir"]
    assert "s_ebarrier_sync" not in kernel.asm["amdgcn"]


@pytest.mark.parametrize("base,worker", [(1, 2), (2, 1), (2, 2)])
def test_multiwave_partition_barrier_has_explicit_diagnostic(base, worker, capfd):
    with pytest.raises(Exception):
        _compile(_multiwave_local_barrier, dict(WORKER=worker), base)
    assert "partition-local barriers require a single wave" in capfd.readouterr().err


@g.jit
def _void_asm(O):
    gl.inline_asm_elementwise("s_nop 0", "", [], dtype=(), is_pure=False, pack=1)
    gl.store(O, 1)


def test_unsupported_resultless_asm_diagnosed_before_result_index(capfd):
    with pytest.raises(Exception):
        _compile(_void_asm, {}, 1)
    assert "HCU elementwise inline asm requires at least one result" in capfd.readouterr().err


@g.jit
def _ordinary_asm(O):
    value = gl.inline_asm_elementwise("v_mov_b32 $0, $1", "=v,v", [7],
                                     dtype=gl.int32, is_pure=False, pack=1)
    gl.store(O, value)


def test_result_asm_still_compiles():
    kernel = _compile(_ordinary_asm, {}, 1)
    assert "v_mov_b32" in kernel.asm["amdgcn"]


def test_single_wave_partition_barrier_changed_replay():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("HCU GPU required")
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != "gfx936":
        pytest.skip("gfx936 GPU required")
    storage = torch.full((512,), -17, device="cuda", dtype=torch.int32)
    out = storage[128:384]
    _local_barrier[(1,)](out, num_warps=1, num_stages=1,
                        terminal_readonly_publication=False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _local_barrier[(1,)](out, num_warps=1, num_stages=1,
                            terminal_readonly_publication=False)
    expected = torch.cat([torch.arange(128, device="cuda"),
                          torch.arange(128, device="cuda") + 1]).int()
    for sentinel in [-1, -7, -123]:
        out.fill_(sentinel)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        assert torch.all(storage[:128] == -17) and torch.all(storage[384:] == -17)


@g.jit
def _partition_shared(A, O, owner, INDEX: gl.constexpr):
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
def _shared_local_barrier(A, O):
    layout: gl.constexpr = gl.SharedLinearLayout(
        [[1], [2], [4], [8], [16], [32], [64]], alignment=512)
    owner = gl.allocate_shared_memory(gl.int32, [2, 128], layout)
    gl.warp_specialize([(_partition_shared, (A, O, owner, 0)),
                        (_partition_shared, (A, O, owner, 1))], [1], [32])
    owner._keep_alive()


def test_single_wave_partition_actual_lds_changed_replay():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("HCU GPU required")
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != "gfx936":
        pytest.skip("gfx936 GPU required")
    a = torch.empty((256,), device="cuda", dtype=torch.int32)
    storage = torch.full((512,), -17, device="cuda", dtype=torch.int32)
    out = storage[128:384]
    options = dict(num_warps=1, num_stages=1, terminal_readonly_publication=False)
    kernel = _shared_local_barrier[(1,)](a, out, **options)
    assert "llvm.amdgcn.wave.barrier" in kernel.asm["llir"]
    assert "ds_write" in kernel.asm["amdgcn"] and "ds_read" in kernel.asm["amdgcn"]
    assert "s_ebarrier_sync" not in kernel.asm["amdgcn"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _shared_local_barrier[(1,)](a, out, **options)
    for seed in [3, 17, 101]:
        a.copy_(torch.arange(256, device="cuda", dtype=torch.int32) * seed - 71)
        before = a.clone()
        out.fill_(-123)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, before) and torch.equal(a, before)
        assert torch.all(storage[:128] == -17) and torch.all(storage[384:] == -17)
