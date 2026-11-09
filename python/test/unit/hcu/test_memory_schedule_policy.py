"""Retained schedule ordering, exact off paths and mandatory native fallback."""
from itertools import product
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

schedule = pytest.importorskip("triton.backends.hcu.memory_schedule")
compiler = pytest.importorskip("triton.backends.hcu.compiler")
# The public backend entrypoint re-exports HIPBackend from compiler_hcu.
# Patch the defining module, where the method resolves its globals.
compiler = import_module(compiler.HIPBackend.__module__)


def options(**overrides):
    values = dict(arch="gfx936", mma_phase_schedule=False,
                  post_mma_wait=False, coalesce_lgkm_waits=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("post,phase,loops", product((False, True), (False, True), (0, 2)))
def test_order_zero_loop_and_exact_off_source(post, phase, loops):
    calls = []

    def wait(source, arch, threads, shared):
        calls.append(("post", source, arch, threads, shared))
        return source + ":wait", loops, "proof result"

    def group(source, arch, warps):
        calls.append(("phase", source, arch, warps))
        return source + ":phase", loops

    metadata = {"shared": 32768}
    policy = schedule.MemorySchedulePolicy.from_options(
        options(post_mma_wait=post, mma_phase_schedule=phase))
    source, prepared = schedule.prepare_memory_schedule(
        "original", metadata, policy,
        SimpleNamespace(prepare_hcu_post_mma_wait=wait, prepare_hcu_mma_phase_loops=group),
        "gfx936", 4)
    expected = "original" + (":wait" if post and loops else "")
    if post:
        assert calls[0] == ("post", "original", "gfx936", 256, 32768)
        assert metadata["hcu_post_mma_wait_loops"] == loops
        assert metadata["hcu_post_mma_wait_preparation_result"] == "proof result"
    if phase and (not post or loops):
        assert calls[-1] == ("phase", expected, "gfx936", 4)
        if loops:
            expected += ":phase"
            assert metadata["hcu_mma_phase_loops"] == loops
    assert source == expected
    assert prepared == bool(loops and (post or phase))
    if not post and not phase:
        assert calls == [] and metadata == {"shared": 32768}


@pytest.mark.parametrize("name", ["mma_phase_schedule", "post_mma_wait", "coalesce_lgkm_waits"])
@pytest.mark.parametrize("invalid", [0, 1, 2, None, "false"])
def test_policy_types(name, invalid):
    with pytest.raises(ValueError, match="must be boolean"):
        schedule.MemorySchedulePolicy.from_options(options(**{name: invalid}))


@pytest.mark.parametrize("value", [0, 1])
def test_sql_boolean_normalization(value):
    from triton.backends.compiler import GPUTarget
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx936", 64))
    names = ("mma_phase_schedule", "post_mma_wait", "coalesce_lgkm_waits")
    parsed = backend.parse_options(dict.fromkeys(names, value))
    assert all(getattr(parsed, name) is bool(value) for name in names)


def test_native_coalescer_target_contract():
    with pytest.raises(ValueError, match="requires gfx936"):
        schedule.MemorySchedulePolicy.from_options(options(arch="gfx942", coalesce_lgkm_waits=True))


def test_only_selected_binding_is_required():
    llvm = SimpleNamespace(prepare_hcu_post_mma_wait=lambda *args: ("prepared", 1, "proof"))
    policy = schedule.MemorySchedulePolicy.from_options(options(post_mma_wait=True))
    assert schedule.prepare_memory_schedule("source", {}, policy, llvm, "gfx936", 4) == ("prepared", True)


def fake_codegen(monkeypatch):
    calls = []
    monkeypatch.setattr(compiler.HIPBackend, "path_to_rocm_clang", staticmethod(lambda: "test-clang"))
    monkeypatch.setattr(compiler.HIPBackend, "_get_clang_args", staticmethod(lambda *_: []))

    def run(argv, **kwargs):
        source = Path(argv[-4]).read_text()
        calls.append(source)
        body = " s_nop 1\n" if "; prepared" in source else " s_nop 0\n"
        Path(argv[-1]).write_text("kernel:\n" + body + " s_endpgm\n")
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("marker", ["hcu.memory.post_mma_wait", "hcu.mma.phase.loop"])
@pytest.mark.parametrize("skip", [False, True])
def test_imported_schedule_native_guard_cannot_be_bypassed(monkeypatch, marker, skip):
    import triton.backends.hcu.mma_phase as native
    calls = fake_codegen(monkeypatch)
    monkeypatch.setattr(native, "check_native_readiness", lambda *_: dict(accepted=False, reason="incomplete operand"))
    source = 'define amdgpu_kernel void @kernel() { ret void }\n; ' + marker
    with pytest.raises(compiler.HSACOError, match="Imported schedule IR failed native completion"):
        compiler.HIPBackend._make_amdgcn_with_schedules(
            source, {}, compiler.HIPOptions(arch="gfx936"), _skip_schedules=skip)
    assert len(calls) == 1


def test_failed_fresh_schedule_recovers_exact_original_source(monkeypatch):
    import triton.backends.hcu.mma_phase as native
    calls = fake_codegen(monkeypatch)

    def prepare(source, metadata, *_):
        metadata.update(hcu_post_mma_wait_loops=1, hcu_mma_phase_loops=1)
        return source + "\n; prepared", True

    monkeypatch.setattr(compiler, "prepare_memory_schedule", prepare)
    monkeypatch.setattr(native, "check_native_readiness", lambda *_: dict(accepted=False, reason="incomplete operand"))
    source = 'define amdgpu_kernel void @kernel() { ret void }'
    metadata = {}
    result = compiler.HIPBackend._make_amdgcn_with_schedules(
        source, metadata, compiler.HIPOptions(arch="gfx936", post_mma_wait=True, mma_phase_schedule=True))
    assert calls == [source + "\n; prepared", source]
    assert result == "kernel:\n s_nop 0\n s_endpgm\n"
    assert metadata["hcu_post_mma_wait_loops"] == metadata["hcu_mma_phase_loops"] == 0
    assert metadata["hcu_post_mma_wait_result"].startswith("fallback: native completion")


@pytest.mark.parametrize("marker", ["hcu.packet.wait.rotation", "hcu.packet.post_mma_wait",
                                  "hcu.packet.operand_overlap.experimental"])
def test_retired_unsafe_imported_schedule_is_rejected(monkeypatch, marker):
    calls = fake_codegen(monkeypatch)
    with pytest.raises(compiler.HSACOError, match="Retired experimental schedule IR"):
        compiler.HIPBackend._make_amdgcn_with_schedules(
            'define amdgpu_kernel void @kernel() { ret void }\n; ' + marker,
            {}, compiler.HIPOptions(arch="gfx936"))
    assert calls == []


@pytest.mark.parametrize("arch,waves,wide,hint,expected", [
    ("gfx936", 8, True, "local-prefetch", True),
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
    backend = compiler.HIPBackend(GPUTarget("hip", arch, 64))
    args = dict(num_warps=waves, prefer_wide_async_copy=wide, schedule_hint=hint)
    parsed = backend.parse_options(args)
    assert parsed.post_mma_wait is expected
    assert parsed.mma_phase_schedule is expected


@pytest.mark.parametrize("manual", [
    {"post_mma_wait": False}, {"mma_phase_schedule": False},
    {"post_mma_wait": True}, {"mma_phase_schedule": True},
    {"post_mma_wait": False, "mma_phase_schedule": True},
    {"post_mma_wait": True, "mma_phase_schedule": False},
    {"post_mma_wait": 0, "mma_phase_schedule": 0},
])
def test_explicit_schedule_options_disable_automatic_pair_resolution(manual):
    from triton.backends.compiler import GPUTarget
    backend = compiler.HIPBackend(GPUTarget("hip", "gfx936", 64))
    parsed = backend.parse_options(dict(num_warps=8, prefer_wide_async_copy=True,
                                        schedule_hint="local-prefetch", **manual))
    assert parsed.post_mma_wait is bool(manual.get("post_mma_wait", False))
    assert parsed.mma_phase_schedule is bool(manual.get("mma_phase_schedule", False))


def test_declined_post_proof_never_runs_phase_alone_for_a_pair():
    def forbidden(*args):
        raise AssertionError("phase-only fallback must not run")
    llvm = SimpleNamespace(prepare_hcu_post_mma_wait=lambda *args: ("unusable", 0, "declined"),
                           prepare_hcu_mma_phase_loops=forbidden)
    policy = schedule.MemorySchedulePolicy(True, True)
    metadata = {"shared": 65536}
    assert schedule.prepare_memory_schedule("original", metadata, policy, llvm, "gfx936", 8) == ("original", False)
