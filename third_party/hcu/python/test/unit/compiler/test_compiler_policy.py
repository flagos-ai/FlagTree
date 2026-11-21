# memory schedule policy
"""Retained schedule ordering, exact off paths and mandatory native fallback."""
from itertools import product
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

_memory_schedule_policy_schedule = pytest.importorskip("triton.backends.hcu.memory_schedule")
_memory_schedule_policy_compiler = pytest.importorskip("triton.backends.hcu.compiler")
# The public backend entrypoint re-exports HIPBackend from compiler_hcu.
# Patch the defining module, where the method resolves its globals.
_memory_schedule_policy_compiler = import_module(_memory_schedule_policy_compiler.HIPBackend.__module__)


@pytest.mark.parametrize("enabled,loops", product((False, True), (0, 2)))
def test_order_zero_loop_and_exact_off_source(enabled, loops):
    calls = []

    def optimize(source, arch, factor, pack, combine, schedule, warps, shared):
        calls.append((source, arch, factor, pack, combine, schedule, warps, shared))
        return "candidate", dict(post_loops=loops, phase_loops=loops, post_reason="proof", factored_stores=0,
                                 packed_bf16_rounds=0, combined_shifts=0)

    metadata = {"shared": 32768}
    options = SimpleNamespace(memory_mma_schedule=enabled, buffer_store_factor_offsets=False, pack_bf16_rounds=False,
                              combine_shifted_disjoint_or=False, arch="gfx942", num_warps=4)
    result = _memory_schedule_policy_schedule.prepare_memory_schedule("original", metadata, options,
                                                                      SimpleNamespace(optimize_hcu_memory_ir=optimize))
    assert result == (("candidate", True) if enabled and loops else ("original", False))
    assert calls == ([("original", "gfx942", False, False, False, True, 4, 32768)] if enabled else [])


@pytest.mark.parametrize("name", ["memory_mma_schedule", "coalesce_lgkm_waits", "combine_shifted_disjoint_or"])
@pytest.mark.parametrize("invalid", [0, 1, 2, None, "false"])
def test_policy_types(name, invalid):
    with pytest.raises(ValueError, match="must be boolean"):
        _memory_schedule_policy_compiler.HIPOptions(arch="gfx936", **{name: invalid})


@pytest.mark.parametrize("value", [0, 1])
def test_sql_boolean_normalization(value):
    from triton.backends.compiler import GPUTarget
    backend = _memory_schedule_policy_compiler.HIPBackend(GPUTarget("hip", "gfx936", 64))
    names = ("memory_mma_schedule", "coalesce_lgkm_waits", "combine_shifted_disjoint_or")
    parsed = backend.parse_options(dict.fromkeys(names, value))
    assert all(getattr(parsed, name) is bool(value) for name in names)


def test_native_coalescer_target_contract():
    with pytest.raises(ValueError, match="requires gfx936"):
        _memory_schedule_policy_compiler.HIPOptions(arch="gfx942", coalesce_lgkm_waits=True)


def _memory_schedule_policy_fake_codegen(monkeypatch):
    calls = []
    monkeypatch.setattr(_memory_schedule_policy_compiler.HIPBackend, "path_to_rocm_clang",
                        staticmethod(lambda: "test-clang"))
    monkeypatch.setattr(_memory_schedule_policy_compiler.HIPBackend, "_get_clang_args", staticmethod(lambda *_: []))

    def run(argv, **kwargs):
        source = Path(argv[-4]).read_text()
        calls.append(source)
        body = " s_nop 1\n" if "; prepared" in source else " s_nop 0\n"
        Path(argv[-1]).write_text("kernel:\n" + body + " s_endpgm\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_memory_schedule_policy_compiler.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("marker", ["hcu.memory.post_mma_wait", "hcu.mma.phase.loop"])
@pytest.mark.parametrize("skip", [False, True])
def test_imported_schedule_native_guard_cannot_be_bypassed(monkeypatch, marker, skip):
    import triton.backends.hcu.mma_phase as native
    calls = _memory_schedule_policy_fake_codegen(monkeypatch)
    monkeypatch.setattr(native, "check_native_readiness", lambda *_: dict(accepted=False, reason="incomplete operand"))
    source = 'define amdgpu_kernel void @kernel() { ret void }\n; ' + marker
    with pytest.raises(_memory_schedule_policy_compiler.HSACOError,
                       match="Imported schedule IR failed native completion"):
        _memory_schedule_policy_compiler.HIPBackend._make_amdgcn_with_schedules(
            source, {}, _memory_schedule_policy_compiler.HIPOptions(arch="gfx936"), _skip_schedules=skip)
    assert len(calls) == 1


def test_failed_fresh_schedule_recovers_exact_original_source(monkeypatch):
    import triton.backends.hcu.mma_phase as native
    calls = _memory_schedule_policy_fake_codegen(monkeypatch)

    def prepare(source, metadata, options, llvm, skip_schedule=False):
        if skip_schedule:
            return source, False
        metadata.update(hcu_post_mma_wait_loops=1, hcu_mma_phase_loops=1)
        return source + "\n; prepared", True

    monkeypatch.setattr(_memory_schedule_policy_compiler, "prepare_memory_schedule", prepare)
    monkeypatch.setattr(native, "check_native_readiness", lambda *_: dict(accepted=False, reason="incomplete operand"))
    source = 'define amdgpu_kernel void @kernel() { ret void }'
    metadata = {}
    result = _memory_schedule_policy_compiler.HIPBackend._make_amdgcn_with_schedules(
        source, metadata, _memory_schedule_policy_compiler.HIPOptions(arch="gfx936", memory_mma_schedule=True))
    assert calls == [source + "\n; prepared", source]
    assert result == "kernel:\n s_nop 0\n s_endpgm\n"
    assert metadata["hcu_post_mma_wait_loops"] == metadata["hcu_mma_phase_loops"] == 0
    assert metadata["hcu_post_mma_wait_result"].startswith("fallback: native completion")


@pytest.mark.parametrize(
    "marker", ["hcu.packet.wait.rotation", "hcu.packet.post_mma_wait", "hcu.packet.operand_overlap.experimental"])
def test_retired_unsafe_imported_schedule_is_rejected(monkeypatch, marker):
    calls = _memory_schedule_policy_fake_codegen(monkeypatch)
    with pytest.raises(_memory_schedule_policy_compiler.HSACOError, match="Retired experimental schedule IR"):
        _memory_schedule_policy_compiler.HIPBackend._make_amdgcn_with_schedules(
            'define amdgpu_kernel void @kernel() { ret void }\n; ' + marker, {},
            _memory_schedule_policy_compiler.HIPOptions(arch="gfx936"))
    assert calls == []


@pytest.mark.parametrize("arch,waves,wide,hint,expected", [
    ("gfx936", 8, True, "local-prefetch", True),
    ("gfx936", 8, True, " attention, LOCAL-PREFETCH , attention ", True),
    ("gfx936", 16, True, "local-prefetch", True),
    ("gfx936", 4, True, "local-prefetch", False),
    ("gfx936", 2, True, "local-prefetch", False),
    ("gfx936", 8, False, "local-prefetch", False),
    ("gfx936", 8, True, "none", False),
    ("gfx936", 8, True, "attention", False),
    ("gfx942", 8, True, "local-prefetch", False),
])
def test_paired_default_uses_target_wave_coverage_and_intent_only(arch, waves, wide, hint, expected):
    from triton.backends.compiler import GPUTarget
    backend = _memory_schedule_policy_compiler.HIPBackend(GPUTarget("hip", arch, 64))
    args = dict(num_warps=waves, prefer_wide_async_copy=wide, schedule_hint=hint)
    parsed = backend.parse_options(args)
    assert parsed.memory_mma_schedule is expected


def test_schedule_hint_normalization_and_explicit_none():
    from triton.backends.compiler import GPUTarget
    backend = _memory_schedule_policy_compiler.HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options({"schedule_hint": " attention, LOCAL-PREFETCH , attention "})
    assert options.schedule_hint == "attention,local-prefetch"
    disabled = backend.parse_options({"schedule_hint": "none", "instruction_sched_variant": "local-prefetch"})
    assert disabled.schedule_hint == "none"


@pytest.mark.parametrize("manual", [False, True, 0, 1])
def test_explicit_schedule_policy_overrides_automatic_default(manual):
    from triton.backends.compiler import GPUTarget
    backend = _memory_schedule_policy_compiler.HIPBackend(GPUTarget("hip", "gfx936", 64))
    parsed = backend.parse_options(
        dict(num_warps=8, prefer_wide_async_copy=True, schedule_hint="local-prefetch", memory_mma_schedule=manual))
    assert parsed.memory_mma_schedule is bool(manual)


# min shared memory
from triton.backends.compiler import GPUTarget
from triton._C import libtriton
if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU backend is not built", allow_module_level=True)
_min_shared_memory_backend = import_module("triton.backends.hcu.compiler")
_min_shared_memory_HIPBackend, _min_shared_memory_HIPOptions = _min_shared_memory_backend.HIPBackend, _min_shared_memory_backend.HIPOptions


@pytest.mark.parametrize('value', [0, 1, 33024, 65536, 1 << 40])
def test_budget_is_offline_resource_intent(value):
    backend = _min_shared_memory_HIPBackend(GPUTarget('hip', 'gfx936', 64))
    options = backend.parse_options(dict(min_shared_memory=value))
    assert options.min_shared_memory == value


@pytest.mark.parametrize('value', [-1, True, False, 1.5, '65536', None])
def test_invalid_budget_rejected_before_launch(value):
    with pytest.raises(ValueError, match='min_shared_memory'):
        _min_shared_memory_HIPOptions(arch='gfx936', min_shared_memory=value)


def test_default_does_not_reserve_memory():
    assert _min_shared_memory_HIPOptions(arch='gfx936').min_shared_memory == 0


# llvm loop unroll
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs
from triton.compiler import ASTSource


@triton.jit
def _llvm_loop_unroll_mm(A, B, C, LICM: tl.constexpr):
    m = tl.arange(0, 64)
    n = tl.arange(0, 128)
    k = tl.arange(0, 32)
    acc = tl.zeros((64, 128), tl.float32)
    for i in tl.range(0, 1024, 32, num_stages=2, disable_licm=LICM):
        a = tle.load(A + m[:, None] * 1024 + (i + k)[None, :], is_async=True)
        b = tle.load(B + (i + k)[:, None] * 128 + n[None, :], is_async=True)
        acc = tl.dot(a, b, acc)
    tl.store(C + m[:, None] * 128 + n[None, :], acc)


class _Captured(Exception):
    pass


@pytest.mark.parametrize('dtype', ['bf16', 'fp16'])
@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('licm', [False, True])
def test_llvm_loop_unroll_metadata(dtype, disable, licm, monkeypatch, tmp_path):
    pytest.importorskip('triton.backends.hcu.compiler_hcu')
    # Find the class actually selected by the backend registry.
    from triton.compiler.compiler import make_backend
    instance = make_backend(GPUTarget('hip', 'gfx936', 64))
    captured = []

    def capture(llir, metadata, options):
        captured.append(llir)
        raise _Captured

    monkeypatch.setattr(knobs.cache, 'dir', str(tmp_path / 'cache'))
    monkeypatch.setattr(knobs.hcu, 'use_buffer_ops', True)
    monkeypatch.setattr(type(instance), 'make_amdgcn', staticmethod(capture))
    source = ASTSource(
        _llvm_loop_unroll_mm,
        signature={p: '*' + dtype
                   for p in ['A', 'B', 'C']},
        constexprs={'LICM': licm},
        attrs={(i, ): [('tt.divisibility', 16), ('tt.pointer_range', 32)]
               for i in range(3)},
    )
    with pytest.raises(_Captured):
        triton.compile(source, target=GPUTarget('hip', 'gfx936', 64),
                       options={'num_warps': 4, 'num_stages': 2, 'disable_llvm_loop_unroll': disable})
    assert len(captured) == 1
    assert ('"llvm.loop.unroll.disable"' in captured[0]) == disable
    assert ('"llvm.licm.disable"' in captured[0]) == licm
    assert instance.parse_options({}).disable_llvm_loop_unroll is False
    assert instance.parse_options({}).hash() != instance.parse_options({'disable_llvm_loop_unroll': True}).hash()


def test_explicit_llvm_unroll_hint_takes_precedence(tmp_path):
    from triton._C.libtriton import hcu, ir
    from triton.compiler.compiler import make_backend
    context = ir.context()
    ir.load_dialects(context)
    make_backend(GPUTarget('hip', 'gfx936', 64)).load_dialects(context)
    source = tmp_path / 'loop.mlir'
    source.write_text('''module {
  tt.func @loop(%end: index) -> i32 {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %value = arith.constant 0 : i32
    %result = scf.for %i = %zero to %end step %one
        iter_args(%acc = %value) -> (i32) {
      scf.yield %acc : i32
    } {llvm.loop_annotation = #llvm.loop_annotation<
        unroll = #llvm.loop_unroll<count = 4 : i32>,
        licm = #llvm.loop_licm<disable = true>>}
    tt.return %result : i32
  }
}
''')
    module = ir.parse_mlir_module(str(source), context)
    module.context = context
    assert module.verify()
    before = module.str()
    hcu.disable_llvm_loop_unroll(module)
    assert module.str() == before


# assume preservation
from triton._C.libtriton import ir, hcu


def test_assumptions_survive_range_folding(tmp_path):
    output = tmp_path / "ir"
    output.mkdir()
    ctx = ir.context()
    ir.load_dialects(ctx)
    hcu.load_dialects(ctx)
    rows = []

    def run(name, body, args='%k: i32', return_type='i32'):
        text = 'module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {\n tt.func public @probe(' + args + ') -> ' + return_type + ' {\n' + body + '\n }\n}'
        f = output / (name + '-before.mlir')
        f.write_text(text)
        mod = ir.parse_mlir_module(str(f), ctx)
        pm = ir.pass_manager(ctx)
        hcu.passes.ttgpuir.add_fold_true_cmpi(pm)
        pm.run(mod, 'C540_assumption_preservation')
        after = str(mod)
        (output / (name + '-after.mlir')).write_text(after)
        row = dict(name=name, after_cmp=after.count('arith.cmpi'),
                   assume_lines=[l.strip() for l in after.splitlines() if 'llvm.intr.assume' in l])
        rows.append(row)
        return after, row

    for name, predicate in [('positive', 'sgt'), ('nonnegative', 'sge'), ('negative', 'slt'),
                            ('unsigned-upper', 'ult')]:
        after, row = run(
            name, '%c0 = arith.constant 7 : i32\n%p = arith.cmpi ' + predicate +
            ', %k, %c0 : i32\nllvm.intr.assume %p : i1\ntt.return %k : i32')
        assert row['after_cmp'] == 1 and len(row['assume_lines']) == 1 and '%true' not in row['assume_lines'][0]
    # The original fact survives while a distinct implied comparison still folds.
    after, row = run(
        'derived-still-folds', '''%c0 = arith.constant 0 : i32
 %a = arith.cmpi sgt, %k, %c0 : i32
 llvm.intr.assume %a : i1
 %derived = arith.cmpi sge, %k, %c0 : i32
 %out = arith.select %derived, %k, %c0 : i32
 tt.return %out : i32''')
    assert row['after_cmp'] == 1 and 'arith.select' not in after and '%true' not in row['assume_lines'][0]
    after, row = run(
        'shared-assume-and-data', '''%c0 = arith.constant 0 : i32
 %a = arith.cmpi sgt, %k, %c0 : i32
 llvm.intr.assume %a : i1
 %out = arith.select %a, %k, %c0 : i32
 tt.return %out : i32''')
    assert row['after_cmp'] == 1 and '%true' not in row['assume_lines'][0]
    after, row = run(
        'unconstrained-control', '''%c0 = arith.constant 0 : i32
 %a = arith.cmpi sgt, %k, %c0 : i32
 %out = arith.select %a, %k, %c0 : i32
 tt.return %out : i32''')
    assert row['after_cmp'] == 1 and 'arith.select' in after and not row['assume_lines']
    # A genuinely constant predicate is allowed to fold independently.
    after, row = run(
        'constant-assume', '''%c0 = arith.constant 0 : i32
 %c1 = arith.constant 1 : i32
 %a = arith.cmpi sgt, %c1, %c0 : i32
 llvm.intr.assume %a : i1
 tt.return %k : i32''')
    assert row['after_cmp'] == 0


@pytest.mark.parametrize("explicit", [None, "none"])
def test_repeated_option_parsing_preserves_caller_input(explicit):
    from triton.backends.compiler import GPUTarget
    backend = _memory_schedule_policy_compiler.HIPBackend(GPUTarget("hip", "gfx936", 64))
    values = {"instruction_sched_variant": "local-prefetch", "enable_mmacfuse": True}
    if explicit is not None:
        values["schedule_hint"] = explicit
    original = dict(values)
    first = backend.parse_options(values)
    second = backend.parse_options(values)
    assert values == original
    assert first.hash() == second.hash()
    assert first.schedule_hint == (explicit or "local-prefetch")
