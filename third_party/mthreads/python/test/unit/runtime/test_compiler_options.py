import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
from triton._C import libtriton
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.experimental.tle.language.gpu.mthreads import MTHREADS_GEMM_COMPILER_OPTIONS_VERSION

if not hasattr(libtriton, "mthreads"):
    pytest.skip("mthreads backend not built in libtriton", allow_module_level=True)


def _get_mthreads_backend():
    if "mthreads" not in backends:
        pytest.skip("mthreads backend not discovered")
    return backends["mthreads"].compiler(GPUTarget("musa", "ph1", 32))


def test_mthreads_can_disable_max_ilp_scheduler():
    from triton.backends.mthreads.compiler import _llc_extra_options

    assert MTHREADS_GEMM_COMPILER_OPTIONS_VERSION >= 1
    backend = _get_mthreads_backend()
    enabled = backend.parse_options({"enable_backend_opt": True})
    assert enabled.disable_max_ilp_scheduler is False
    enabled_args = _llc_extra_options({"uses_sqmma": True}, enabled, capability=31)
    assert "-misched=mtgpu-max-ilp" in enabled_args

    disabled = backend.parse_options({
        "enable_backend_opt": True,
        "disable_max_ilp_scheduler": True,
    })
    disabled_args = _llc_extra_options({"uses_sqmma": True}, disabled, capability=31)
    assert "-misched=mtgpu-max-ilp" not in disabled_args
    assert "-mtgpu-opt-level=1" in disabled_args
    assert disabled.hash() != enabled.hash()


def test_mthreads_disable_max_ilp_scheduler_does_not_change_non_sqmma_options():
    from triton.backends.mthreads.compiler import _llc_extra_options

    backend = _get_mthreads_backend()
    enabled = backend.parse_options({"enable_backend_opt": True})
    disabled = backend.parse_options({
        "enable_backend_opt": True,
        "disable_max_ilp_scheduler": True,
    })

    enabled_args = _llc_extra_options({"uses_sqmma": False}, enabled, capability=31)
    disabled_args = _llc_extra_options({"uses_sqmma": False}, disabled, capability=31)
    assert disabled_args == enabled_args


def test_mthreads_user_scheduler_and_opt_level_replace_backend_defaults():
    from triton.backends.mthreads.compiler import _llc_extra_options

    backend = _get_mthreads_backend()
    options = backend.parse_options({
        "enable_backend_opt": True,
        "llc_options": "-misched=mtgpu-max-occupancy -mtgpu-opt-level=2",
    })
    args = _llc_extra_options({"uses_sqmma": True}, options, capability=31)

    assert args.count("-misched=mtgpu-max-occupancy") == 1
    assert "-misched=mtgpu-max-ilp" not in args
    assert args.count("-mtgpu-opt-level=2") == 1
    assert "-mtgpu-opt-level=1" not in args


def test_mthreads_user_llc_pass_manager_level_replaces_default():
    from triton.backends.mthreads.compiler import _llc_extra_options, _llc_opt_level

    backend = _get_mthreads_backend()
    default = backend.parse_options({})
    assert _llc_opt_level(default) == "-O2"

    options = backend.parse_options({"llc_options": "-O3 -O1"})
    assert _llc_opt_level(options) == "-O1"
    args = _llc_extra_options({"uses_sqmma": True}, options, capability=31)
    assert not any(arg in ("-O0", "-O1", "-O2", "-O3") for arg in args)


@pytest.mark.parametrize("tool_name", ["llc", "llc-asm"])
def test_mthreads_known_llc_register_failure_is_recoverable(monkeypatch, tmp_path, tool_name):
    from triton.backends.mthreads import compiler as musa_compiler
    from triton.runtime.errors import PTXASError

    assert musa_compiler.MTHREADS_REGISTER_FAILURE_RECOVERY_VERSION >= 1

    class FailedProcess:
        returncode = 134
        stdout = ""
        stderr = "LLVM ERROR: no registers from class available to allocate"

    monkeypatch.setattr(musa_compiler.subprocess, "run", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(PTXASError, match="MTGPU register allocation"):
        musa_compiler._run_tool_command(tool_name, [tool_name, "kernel.ll", "-O2"], repro_dir=tmp_path)


@pytest.mark.parametrize("tool_name", ["llc", "llc-asm"])
def test_mthreads_unknown_llc_failure_is_not_swallowed(monkeypatch, tmp_path, tool_name):
    from triton.backends.mthreads import compiler as musa_compiler

    class FailedProcess:
        returncode = 134
        stdout = ""
        stderr = "LLVM ERROR: unexpected backend failure"

    monkeypatch.setattr(musa_compiler.subprocess, "run", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(RuntimeError, match=rf"`{tool_name}` failed with error code 134"):
        musa_compiler._run_tool_command(tool_name, [tool_name, "kernel.ll", "-O2"], repro_dir=tmp_path)


def test_mthreads_post_ra_register_crash_is_recoverable(monkeypatch, tmp_path):
    from triton.backends.mthreads import compiler as musa_compiler
    from triton.runtime.errors import PTXASError

    class FailedProcess:
        returncode = 139
        stdout = ""
        stderr = "MTGPU Post-RA Internal Registers Related Optimization"

    monkeypatch.setattr(musa_compiler.subprocess, "run", lambda *args, **kwargs: FailedProcess())
    with pytest.raises(PTXASError, match="MTGPU register allocation"):
        musa_compiler._run_tool_command("llc", ["llc", "kernel.ll", "-O0"], repro_dir=tmp_path)


@pytest.mark.parametrize(
    ("arch", "expected_capability", "expected_arch"),
    [
        ("ph1", 31, "31"),
        ("ph1s", 32, "32"),
        ("mp31", 31, "31"),
        ("mp_31", 31, "31"),
        ("mp32", 32, "32"),
        ("mp_32", 32, "32"),
    ],
)
def test_mthreads_arch_alias_capability(arch, expected_capability, expected_arch):
    from triton.backends.mthreads._musa_arch import musa_capability_from_arch, normalize_musa_arch
    from triton.backends.mthreads.compiler import _capability_from_arch
    from triton.backends.mthreads.driver import _arch_to_musa_capability

    assert normalize_musa_arch(arch) == expected_arch
    assert musa_capability_from_arch(arch) == expected_capability
    assert _capability_from_arch(arch) == expected_capability
    assert _arch_to_musa_capability(arch) == expected_capability

    backend = _get_mthreads_backend()
    assert backend.parse_options({"arch": arch}).arch == expected_arch


def test_mthreads_rejects_unavailable_future_architecture():
    backend = _get_mthreads_backend()
    with pytest.raises(ValueError, match="supports capabilities up to 32"):
        backend.parse_options({"arch": "33"})


def test_mthreads_can_disable_sqmma_m_split_preference():
    assert MTHREADS_GEMM_COMPILER_OPTIONS_VERSION >= 2
    backend = _get_mthreads_backend()
    enabled = backend.parse_options({})
    disabled = backend.parse_options({"disable_sqmma_m_split_preference": True})

    assert enabled.disable_sqmma_m_split_preference is False
    assert disabled.disable_sqmma_m_split_preference is True
    assert disabled.hash() != enabled.hash()


def test_mthreads_k128_sqmma_environment_is_part_of_cache_key(monkeypatch):
    backend = _get_mthreads_backend()
    monkeypatch.delenv("TRITON_MUSA_ENABLE_K128_SQMMA", raising=False)
    disabled_hash = backend.parse_options({}).hash()
    monkeypatch.setenv("TRITON_MUSA_ENABLE_K128_SQMMA", "10")
    invalid_hash = backend.parse_options({}).hash()
    monkeypatch.setenv("TRITON_MUSA_ENABLE_K128_SQMMA", "1")
    enabled_hash = backend.parse_options({}).hash()

    assert invalid_hash == disabled_hash
    assert enabled_hash != disabled_hash


def test_mthreads_rejects_ineffective_per_kernel_k128_option():
    backend = _get_mthreads_backend()

    with pytest.raises(ValueError, match="not a per-kernel MThreads option"):
        backend.parse_options({"k128_sqmma_opt_in": "1"})
