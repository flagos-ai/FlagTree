from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import triton
import triton.language as tl


@pytest.mark.parametrize("env", [None, "0", "1", "false", "true", "TRUE"])
@pytest.mark.parametrize("option", [None, False, True])
@pytest.mark.parametrize("pipeline", ["a2_a3", "910_95", "simt"])
def test_auto_blockify_compile_option(monkeypatch, env, option, pipeline):
    """Inspect the real command builders without invoking BiSheng or an NPU."""
    compiler = pytest.importorskip("triton.backends.ascend.compiler")
    if env is None:
        monkeypatch.delenv("TRITON_ALL_BLOCKS_PARALLEL", raising=False)
    else:
        monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", env)
    options = compiler.NPUOptions(enable_auto_blockify=option, force_simt_only=pipeline == "simt")
    metadata = asdict(options)
    metadata.update(target=SimpleNamespace(arch="Ascend910B3"), bitcodes=[])
    monkeypatch.setattr(compiler, "_parse_linalg_metadata", lambda mod, meta: (mod, meta))
    monkeypatch.setattr(compiler, "_parse_ttir_metadata", lambda mod, meta: meta)
    monkeypatch.setattr(compiler, "_check_bishengir_api_change", lambda: True)
    monkeypatch.setattr(compiler, "_check_bishengir_is_regbased", lambda: False)
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("bishengir-compile", {}))
    monkeypatch.setattr(compiler, "NPUUtils", lambda: SimpleNamespace(get_arch=lambda: "Ascend910B3"))
    # The A2/A3 builder also copies its input to a debug path outside tmpdir.
    monkeypatch.setattr("shutil.copy2", lambda *args: None)
    commands = []

    def compile_binary(command, **kwargs):
        commands.append(command)
        Path(command[command.index("-o") + 1] + ".o").write_bytes(b"binary")
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    monkeypatch.setattr(compiler.subprocess, "run", compile_binary)
    builders = {
        "a2_a3": compiler.linalg_to_bin_enable_npu_compile_A2_A3,
        "910_95": compiler.linalg_to_bin_enable_npu_compile_910_95,
        "simt": compiler.ttir_to_npubin,
    }
    assert builders[pipeline]("module {}", metadata, options) == b"binary"
    expected = option if option is not None else env in ("1", "true", "TRUE")
    assert len(commands) == 1
    assert commands[0].count("--enable-auto-blockify-loop") == int(expected)


@pytest.mark.parametrize("env", [None, "0", "1"])
@pytest.mark.parametrize("option", [None, False, True])
def test_each_program_executes_once(device, monkeypatch, fresh_triton_cache, env, option):
    """Catch both repeated programs (opt-out) and missing programs (opt-in)."""
    if device != "npu":
        pytest.skip("requires an Ascend NPU; run with --device=npu")
    import torch
    import torch_npu  # noqa: F401
    from triton.backends.ascend.driver import NPUUtils

    if env is None:
        monkeypatch.delenv("TRITON_ALL_BLOCKS_PARALLEL", raising=False)
    else:
        monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", env)

    # A new JIT object and a fresh disk cache isolate each ENV/option pair.
    @triton.jit
    def count_programs(counts, seen, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(seen + offsets, offsets)
        tl.atomic_add(counts + pid, 1, sem="relaxed")

    programs = 2 * NPUUtils().get_aivector_core_num() + 1
    counts = torch.zeros(programs, device=device, dtype=torch.int32)
    seen = torch.full((programs * 128, ), -1, device=device, dtype=torch.int32)
    kwargs = {} if option is None else {"enable_auto_blockify": option}
    count_programs[(programs, )](counts, seen, BLOCK=128, mix_mode="aiv", **kwargs)
    torch.npu.synchronize()
    assert torch.equal(counts.cpu(), torch.ones(programs, dtype=torch.int32))
    assert torch.equal(seen.cpu(), torch.arange(programs * 128, dtype=torch.int32))
