"""HCU lowering tests that do not initialize a GPU or run the vendor assembler."""
import re

import pytest

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

pytest.importorskip("triton.backends.hcu.compiler_hcu")
from triton.backends.hcu.compiler_hcu import HIPBackend


@triton.jit
def _matmul(A, B, C, ASYNC_A: tl.constexpr, ASYNC_B: tl.constexpr,
            BLOCK_PTR: tl.constexpr, STAGES: tl.constexpr):
    m = tl.arange(0, 64)
    n = tl.arange(0, 64)
    k = tl.arange(0, 64)
    if BLOCK_PTR:
        ap = tl.make_block_ptr(A, (1024, 1024), (1024, 1), (0, 0), (64, 64), (1, 0))
        bp = tl.make_block_ptr(B, (1024, 1024), (1024, 1), (0, 0), (64, 64), (1, 0))
    acc = tl.zeros((64, 64), tl.float32)
    for i in tl.range(0, 16, num_stages=STAGES):
        if BLOCK_PTR:
            if ASYNC_A is None:
                a = tl.load(ap, boundary_check=(0, 1), padding_option="zero")
            else:
                a = tle.load(ap, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC_A)
            if ASYNC_B is None:
                b = tl.load(bp, boundary_check=(0, 1), padding_option="zero")
            else:
                b = tle.load(bp, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC_B)
            ap = tl.advance(ap, (0, 64))
            bp = tl.advance(bp, (64, 0))
        else:
            a_ptr = A + m[:, None] * 1024 + (i * 64 + k[None, :])
            b_ptr = B + (i * 64 + k[:, None]) * 1024 + n[None, :]
            if ASYNC_A is None:
                a = tl.load(a_ptr)
            else:
                a = tle.load(a_ptr, is_async=ASYNC_A)
            if ASYNC_B is None:
                b = tl.load(b_ptr)
            else:
                b = tle.load(b_ptr, is_async=ASYNC_B)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * 1024 + n[None, :], acc)


class _LoweringFinished(Exception):
    pass


@triton.jit
def _local_copy(A, C, POLICY: tl.constexpr, COPY: tl.constexpr):
    r = tl.broadcast_to(tl.arange(0, 16)[:, None], (16, 64))
    c = tl.broadcast_to(tl.arange(0, 64)[None, :], (16, 64))
    buf = tle.gpu.alloc((16, 64), dtype=tl.bfloat16, nv_mma_shared_layout=False)
    if COPY:
        tle.gpu.copy(A + r * 64 + c, buf, (16, 64))
    else:
        if POLICY == -1:
            value = tl.load(A + r * 64 + c)
        else:
            value = tle.load(A + r * 64 + c, is_async=POLICY == 1)
        buf.store(value)
    tl.store(C + r * 64 + c, buf.load())


def _ttgir(monkeypatch, async_a, async_b, block_ptr, **options):
    captured = []

    def capture(mod, metadata, opts):
        captured.append(str(mod))
        raise _LoweringFinished

    monkeypatch.setattr(HIPBackend, "make_llir", staticmethod(capture))
    source = ASTSource(
        _matmul, signature={"A": "*bf16", "B": "*bf16", "C": "*bf16"},
        constexprs={"ASYNC_A": async_a, "ASYNC_B": async_b, "BLOCK_PTR": block_ptr,
                    "STAGES": options.get("num_stages", 2)},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(3)},
    )
    with pytest.raises(_LoweringFinished):
        triton.compile(source, target=GPUTarget("hip", options.pop("arch", "gfx936"), 64),
                       options={"num_warps": 4, "num_stages": 2, "optimize_epilogue": False, **options})
    return captured[0]


@pytest.mark.parametrize("global_async", [False, True])
@pytest.mark.parametrize("per_op", [None, False, True])
@pytest.mark.parametrize("block_ptr", [False, True])
def test_async_load_override(monkeypatch, global_async, per_op, block_ptr):
    monkeypatch.setattr(knobs.hcu, "use_async_copy", global_async)
    ir = _ttgir(monkeypatch, per_op, per_op, block_ptr)
    has_async = "async_copy_global_to_local" in ir or "buffer_load_to_local" in ir
    assert has_async == (global_async if per_op is None else per_op)


@pytest.mark.parametrize("global_async", [False, True])
@pytest.mark.parametrize("other_policy", [None, False])
def test_mixed_load_policies(monkeypatch, global_async, other_policy):
    monkeypatch.setattr(knobs.hcu, "use_async_copy", global_async)
    ir = _ttgir(monkeypatch, True, other_policy, True)
    assert "async_copy_global_to_local" in ir or "buffer_load_to_local" in ir
    has_sync = "tt.load " in ir or "buffer_load " in ir
    assert has_sync == (other_policy is False or not global_async)


@pytest.mark.parametrize("block_ptr", [False, True])
@pytest.mark.parametrize("global_async", [False, True])
def test_pipelined_loads_are_not_copied_twice(monkeypatch, block_ptr, global_async):
    monkeypatch.setattr(knobs.hcu, "use_async_copy", global_async)
    ir = _ttgir(monkeypatch, True, True, block_ptr)
    # Two input tiles in the prologue and two in the loop. A leftover dead
    # tt.load must not become another side-effecting copy in TLE lowering.
    copies = ir.count("async_copy_global_to_local") + ir.count("buffer_load_to_local")
    assert copies == 4


@pytest.mark.parametrize("copy,policy", [(False, -1), (False, 0), (False, 1), (True, -1)])
def test_global_to_local_store_policy(monkeypatch, copy, policy):
    captured = []

    def capture(mod, metadata, opts):
        captured.append(str(mod))
        raise _LoweringFinished

    monkeypatch.setattr(HIPBackend, "make_llir", staticmethod(capture))
    source = ASTSource(
        _local_copy, signature={"A": "*bf16", "C": "*bf16"},
        constexprs={"POLICY": policy, "COPY": copy},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(2)},
    )
    with pytest.raises(_LoweringFinished):
        triton.compile(source, target=GPUTarget("hip", "gfx936", 64), options={"num_stages": 1})
    ir = captured[0]
    copies = ir.count("async_copy_global_to_local") + ir.count("buffer_load_to_local")
    assert copies == (0 if policy == 0 else 1)
    assert ir.count("ttg.local_alloc") == 1


@pytest.mark.parametrize("layout", [2, 3, 4])
def test_unsupported_mmac_layout(monkeypatch, layout, capfd):
    with pytest.raises(RuntimeError):
        _ttgir(monkeypatch, False, False, False, mmac_layout_force=layout)
    assert "require lit/lts support" in capfd.readouterr().err


@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
@pytest.mark.parametrize("policy", [False, True])
def test_legacy_matrix_read_selection(monkeypatch, arch, policy):
    captured = []

    def capture(src, metadata, options):
        captured.append(src)
        raise _LoweringFinished

    monkeypatch.setattr(HIPBackend, "make_amdgcn", staticmethod(capture))
    source = ASTSource(
        _matmul, signature={"A": "*bf16", "B": "*bf16", "C": "*bf16"},
        constexprs={"ASYNC_A": policy, "ASYNC_B": policy, "BLOCK_PTR": False,
                    "STAGES": 2},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(3)},
    )
    with pytest.raises(_LoweringFinished):
        triton.compile(source, target=GPUTarget("hip", arch, 64),
                       options={"num_warps": 4, "num_stages": 2,
                                "optimize_epilogue": False})
    assert ("llvm.hcu.ds.read.m32x16.b16" in captured[0]) == (arch == "gfx936")
    if arch == "gfx936" and policy:
        calls = [line for line in captured[0].splitlines()
                 if "call " in line and "llvm.hcu.ds.read.m32x16.b16" in line]
        assert any("!noalias" in line for line in calls)


@pytest.mark.parametrize("warp_m", [1, 2, 4])
def test_mmac_wave_partition(monkeypatch, warp_m):
    ir = _ttgir(monkeypatch, False, False, False, mmac_warps_m=warp_m)
    assert f'#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [{warp_m}, {4 // warp_m}]' in ir


def test_mmac_wave_partition_geometry(monkeypatch, capfd):
    with pytest.raises(RuntimeError):
        _ttgir(monkeypatch, False, False, False, num_warps=8, mmac_warps_m=8)
    assert "does not fit" in capfd.readouterr().err


@pytest.mark.parametrize("arch,warps,m", [("gfx936", 8, 3), ("gfx936", 4, 8), ("gfx938", 8, 4)])
def test_mmac_wave_partition_options(arch, warps, m):
    from triton.backends.hcu.compiler_hcu import HIPOptions
    with pytest.raises(ValueError, match="mmac_warps_m"):
        HIPOptions(arch=arch, num_warps=warps, mmac_warps_m=m)


@triton.jit
def _pingpong_matmul(A, B, C, STAGES: tl.constexpr):
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    k = tl.arange(0, 32)
    acc = tl.zeros((128, 128), tl.float32)
    for i in tl.range(0, 32, num_stages=STAGES):
        rk = i * 32 + k
        a = tle.load(A + m[:, None] * 1024 + rk[None, :],
                     (m[:, None] < 1024) & (rk[None, :] < 1024),
                     0, is_async=True)
        b = tle.load(B + rk[:, None] * 1024 + n[None, :],
                     (rk[:, None] < 1024) & (n[None, :] < 1024),
                     0, is_async=True)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * 1024 + n[None, :], acc,
             (m[:, None] < 1024) & (n[None, :] < 1024))


@pytest.mark.parametrize("stages", [3, 4])
@pytest.mark.parametrize("global_async", [False, True])
def test_async_pingpong_requires_tail_wait(monkeypatch, stages, global_async):
    captured = []

    def capture(mod, metadata, options):
        captured.append(str(mod))
        raise _LoweringFinished

    monkeypatch.setattr(knobs.hcu, "use_block_pingpong", True)
    monkeypatch.setattr(knobs.hcu, "use_async_copy", global_async)
    monkeypatch.setattr(HIPBackend, "make_llir", staticmethod(capture))
    source = ASTSource(
        _pingpong_matmul,
        signature={"A": "*bf16", "B": "*bf16", "C": "*bf16"},
        constexprs={"STAGES": stages},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)]
               for i in range(3)},
    )
    # TTGIR verification must succeed without initializing a GPU. In the
    # four-stage case the old transform moved local_load above its wait token.
    with pytest.raises(_LoweringFinished):
        triton.compile(source, target=GPUTarget("hip", "gfx936", 64),
                       options={"num_warps": 8, "num_stages": stages,
                                "optimize_epilogue": False})
    ir = captured[0]
    assert "buffer_load_to_local" in ir or "async_copy_global_to_local" in ir
    # Preserve the supported three-stage optimization; four stages retain
    # their ordinary valid software pipeline without asymmetric pingpong.
    assert ("hcug.cond_barrier" in ir) == (stages == 3)


def _dot_inputs_are_loop_carried(ir):
    header = next(line for line in ir.splitlines() if "scf.for " in line)
    args = header.split("iter_args(", 1)[1].split(") ->", 1)[0]
    carried = set(re.findall(r"(%[\w.$]+)\s*=", args))
    dot = next(line for line in ir.splitlines() if "tt.dot " in line)
    operands = re.search(r"tt.dot\s+(%[\w.$]+),\s*(%[\w.$]+),", dot)
    return all(value in carried for value in operands.groups())


@pytest.mark.parametrize("global_async", [False, True])
@pytest.mark.parametrize("block_ptr", [False, True])
def test_local_prefetch_carries_dot_operands(monkeypatch, global_async, block_ptr):
    monkeypatch.setattr(knobs.hcu, "use_async_copy", global_async)
    monkeypatch.setattr(knobs.hcu, "use_block_pingpong", False)
    # Explicit synchronous policies must override even a global async default.
    baseline = _ttgir(monkeypatch, False, False, block_ptr)
    prefetched = _ttgir(monkeypatch, False, False, block_ptr,
                        schedule_hint="local-prefetch")
    assert not _dot_inputs_are_loop_carried(baseline)
    assert _dot_inputs_are_loop_carried(prefetched)
    # Both paths use one LDS tile per operand. The opt-in adds VGPR prefetch,
    # rather than increasing the shared allocation to two buffers.
    allocs = [line for line in prefetched.splitlines() if "ttg.local_alloc " in line]
    assert len(allocs) == 2
    assert all(re.search(r"!ttg.memdesc<(?:1x)?64x64xbf16,", line) for line in allocs)


@pytest.mark.parametrize("arch,stages,policies", [
    ("gfx936", 2, (True, True)),
    ("gfx936", 2, (True, False)),
    ("gfx936", 2, (False, True)),
    ("gfx936", 4, (False, False)),
    ("gfx938", 2, (False, False)),
])
def test_local_prefetch_preserves_unsupported_paths(monkeypatch, arch, stages, policies):
    monkeypatch.setattr(knobs.hcu, "use_async_copy", False)
    monkeypatch.setattr(knobs.hcu, "use_block_pingpong", False)
    baseline = _ttgir(monkeypatch, *policies, False, arch=arch, num_stages=stages)
    prefetched = _ttgir(monkeypatch, *policies, False, arch=arch, num_stages=stages,
                        schedule_hint="local-prefetch")
    # Compare generated operations, not merely the presence of the opt-in attr.
    prefetched = re.sub(r'"?ttg\.hcu\.local_prefetch"?\s*=\s*1\s*:\s*i32,?\s*',
                        "", prefetched)
    assert prefetched == baseline


@pytest.mark.parametrize("num_warps,stages", [(4, 2), (4, 3), (4, 4), (8, 2)])
@pytest.mark.parametrize("global_async", [False, True])
def test_async_pingpong_copy_supply_outside_transform_domain(
    monkeypatch, num_warps, stages, global_async
):
    captured = []

    def capture(mod, metadata, options):
        captured.append(str(mod))
        raise _LoweringFinished

    monkeypatch.setattr(knobs.hcu, "use_buffer_ops", True)
    monkeypatch.setattr(knobs.hcu, "use_block_pingpong", True)
    monkeypatch.setattr(knobs.hcu, "use_async_copy", global_async)
    monkeypatch.setattr(HIPBackend, "make_llir", staticmethod(capture))
    source = ASTSource(
        _pingpong_matmul,
        signature={"A": "*bf16", "B": "*bf16", "C": "*bf16"},
        constexprs={"STAGES": stages},
        attrs={(i,): [("tt.divisibility", 16), ("tt.pointer_range", 32)] for i in range(3)},
    )
    with pytest.raises(_LoweringFinished):
        triton.compile(
            source,
            target=GPUTarget("hip", "gfx936", 64),
            options={"num_warps": num_warps, "num_stages": stages, "optimize_epilogue": False},
        )
    assert len(captured) == 1
    assert "buffer_load_to_local" in captured[0]
    assert "hcug.cond_barrier" not in captured[0]
