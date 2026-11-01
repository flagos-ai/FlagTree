# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega-on-Radeon end-to-end: a real Qwen3 bf16 model compiles through the
amd-gfx1100/gfx1201 target, renders an RDNA megakernel (tile core / pipe / warp
specialize), runs on an AMD RDNA GPU, and matches the torch evaluator.

Both compile paths are exercised: the plain ``Compiler(target=...)`` default
distribution, and the ``amd-rdna`` recipe path, which supplies its own
implementation model *through the target*. The two reach the compiler
differently, so testing only the default path would miss regressions on the
recipe path (e.g. a supplied model bypassing the cooperative warp cap)."""

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.evaluator import (
    CheckpointWeightResolver,
    TorchEvaluator,
    create_paged_attention_state,
)
from triton.flagmega.importer import import_qwen3_model
from triton.flagmega.runtime import load as load_runtime

from .helpers import full_checkpoint

torch = pytest.importorskip("torch")


def _amd_rdna_target():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        return None
    # Read the real gfx arch; gcnArchName may carry feature suffixes, e.g.
    # "gfx1100:sramecc+:xnack-". Map only known RDNA parts -- an unknown arch
    # returns None so the test skips rather than silently compiling for the
    # wrong target.
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "").split(":")[0]
    return {"gfx1100": "amd-gfx1100", "gfx1201": "amd-gfx1201"}.get(arch)


def _compiler_for(target_name, *, supplied_model):
    """Build the Compiler for an AMD target. When ``supplied_model`` is set, feed
    an implementation model *through the target* -- the path the amd-rdna recipe
    uses -- instead of letting the machine build its own."""
    compiler = Compiler(CompileOptions(target=target_name))
    if supplied_model:
        from triton.flagmega.targets.amd_gfx1100 import AmdGfx1100Target
        from triton.flagmega.targets.amd_gfx1201 import AmdGfx1201Target
        from triton.flagmega.targets.portable_triton_implementations import (
            portable_triton_implementation_model,
        )

        target_cls = {"amd-gfx1100": AmdGfx1100Target, "amd-gfx1201": AmdGfx1201Target}[target_name]
        compiler.target = target_cls(triton_implementation_model=portable_triton_implementation_model())
    return compiler


@pytest.mark.parametrize(
    "supplied_model", [False, True], ids=["default-distribution", "recipe-supplied-model"]
)
def test_two_layer_qwen3_runs_on_amd_rdna(tmp_path, supplied_model):
    target = _amd_rdna_target()
    if target is None:
        pytest.skip("AMD RDNA (ROCm/HIP) GPU is required")

    checkpoint = full_checkpoint(num_hidden_layers=2)
    imported = import_qwen3_model(checkpoint)
    compiled = _compiler_for(target, supplied_model=supplied_model).compile(imported).module
    assert compiled.stage == "bufferized_tir"

    artifact = write_artifact(
        compiled,
        tmp_path / "qwen3-amd",
        target=target,
        checkpoint=checkpoint,
        emit_executable=True,
    )
    runtime = load_runtime(artifact, device="cuda:0")
    runtime._pool_values["workspace"].fill_(0xA5)

    state = runtime.create_state()
    reference_state = create_paged_attention_state(runtime.state_config, device="cuda:0")
    input_ids = torch.tensor([3], dtype=torch.int32, device="cuda:0")

    evaluator = TorchEvaluator(CheckpointWeightResolver(checkpoint, device="cuda:0"))
    expected_logits, expected_token, _ = evaluator.run(
        imported,
        {"input_ids": input_ids, "paged_attention_kv_cache": reference_state},
    )

    logits, next_token = runtime.create_outputs()
    runtime.prepare(input_ids, state, logits=logits, next_token=next_token)
    runtime.run_into(logits, next_token, input_ids, state)
    torch.cuda.synchronize()

    torch.testing.assert_close(logits, expected_logits, atol=0.15, rtol=0.04)
    assert next_token.item() == expected_token.item()
    # gfx1201 compiles the megakernel without buffer ops: its small-phase loads
    # would otherwise become serial waterfall loops over VGPR base pointers.
    if target == "amd-gfx1201":
        for prepared in runtime._prepared:
            assert "buffer_load" not in prepared.compiled_kernel.asm["amdgcn"]
