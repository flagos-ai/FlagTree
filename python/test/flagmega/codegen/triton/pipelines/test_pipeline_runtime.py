# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import re

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.ir import DType
from triton.flagmega.runtime import load


def _assert_reused_device_roles(runtime, *, wrapper_depth=0):
    llir = runtime._prepared.compiled_kernel.asm["llir"]
    assert not re.search(r"^define .*@[^ (]*\.flagmega_main__(?:consumer|producer)__", llir, re.M)
    names = ["worker", *(f"wrapper_{index}" for index in range(wrapper_depth))]
    for name in names:
        for role in ("consumer", "producer"):
            definitions = re.findall(
                r"^define .*@([^ (]*\._flagmega_function_" + name + "__" + role + r"__[^ (]*)\(",
                llir, re.M,
            )
            assert len(definitions) == 1, (name, role, definitions)
            expected_calls = 2 if name == names[-1] else 1
            assert len(re.findall(r"\bcall\b[^\n]*@" + re.escape(definitions[0]) + r"\(", llir)) == expected_calls


def _checkpoint(weights):
    return MemoryCheckpoint(
        {},
        {
            name: TensorInfo(name, DType.BFLOAT16, (128, 128), "memory")
            for name in weights
        },
        weights,
    )


def test_reusable_pipeline_matches_torch_on_sm90(
    tmp_path, compile_pipeline_module,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")

    first_weight = torch.randn(
        (128, 128), dtype=torch.bfloat16
    ) / 8
    second_weight = torch.randn(
        (128, 128), dtype=torch.bfloat16
    ) / 8
    artifact = write_artifact(
        compile_pipeline_module(reusable=True),
        tmp_path / "dense-pipeline",
        target="nvidia-sm90",
        checkpoint=_checkpoint({
            "first_weight": first_weight,
            "second_weight": second_weight,
        }),
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((1, 128), dtype=torch.bfloat16, device="cuda:0")
    output = torch.empty_like(value)

    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()

    expected = (value @ first_weight.to("cuda:0").T) @ second_weight.to("cuda:0").T
    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    assert runtime.resource_report["shared_memory_bytes"] >= 8192
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
    _assert_reused_device_roles(runtime)


@pytest.mark.parametrize("wrapper_depth", [0, 1])
def test_reusable_multi_pipeline_drains_each_endpoint_once_on_sm90(
    tmp_path, compile_pipeline_module, wrapper_depth,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")

    weights = tuple(
        torch.randn((128, 128), dtype=torch.bfloat16) / 8
        for _ in range(4)
    )
    checkpoint_weights = {
        **{f"first_weight_{index}": weights[index] for index in range(2)},
        **{f"second_weight_{index}": weights[index + 2] for index in range(2)},
    }
    artifact = write_artifact(
        compile_pipeline_module(reusable=True, worker_depth=2, wrapper_depth=wrapper_depth),
        tmp_path / "dense-multi-pipeline",
        target="nvidia-sm90",
        checkpoint=_checkpoint(checkpoint_weights),
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    value = torch.randn((1, 128), dtype=torch.bfloat16, device="cuda:0")
    output = torch.empty_like(value)

    runtime.prepare(value, output=output)
    runtime.run_into(output, value)
    torch.cuda.synchronize()

    expected = value
    for weight in weights:
        expected = expected @ weight.to("cuda:0").T
    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0
    torch.testing.assert_close(output, expected, rtol=4e-2, atol=4e-2)
    _assert_reused_device_roles(runtime, wrapper_depth=wrapper_depth)
