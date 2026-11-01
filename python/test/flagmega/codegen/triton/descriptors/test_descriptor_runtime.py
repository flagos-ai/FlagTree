# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
# Runtime integration for compiler-produced tensor descriptors.

import pytest

from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load

@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("dag", [False, True])
def test_reused_descriptor_gemv_matches_torch_on_sm90(
    tmp_path, transpose_b, dag, two_call_descriptor_module,
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")

    artifact = write_artifact(
        two_call_descriptor_module(transpose_b=transpose_b, program_dag=dag),
        tmp_path / "descriptor-gemv",
        target="nvidia-sm90",
        emit_executable=True,
    )
    runtime = load(artifact, device="cuda:0")
    stream = torch.cuda.Stream() if dag else None
    value = torch.randn((1, 128), dtype=torch.bfloat16, device="cuda:0")
    first_weight = torch.randn(
        (128, 128), dtype=torch.bfloat16, device="cuda:0") / 8
    second_weight = torch.randn(
        (128, 128), dtype=torch.bfloat16, device="cuda:0") / 8
    output = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda:0")

    runtime.prepare(value, first_weight, second_weight, output=output)
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
    runtime.run_into(output, value, first_weight, second_weight, stream=stream.cuda_stream if stream is not None else None)
    torch.cuda.synchronize()

    first_rhs = first_weight.T if transpose_b else first_weight
    second_rhs = second_weight.T if transpose_b else second_weight
    expected = value @ first_rhs @ second_rhs
    assert runtime.prepare_count == 1
    if dag:
        first, second = [i for i, spec in enumerate(runtime.codegen["kernels"]) if spec["entry_function"] == "worker"]
        assert runtime.kernels[first] is runtime.kernels[second]
        assert runtime._prepared[first].compiled_kernel is runtime._prepared[second].compiled_kernel
        assert all(r["spill_bytes"] == 0 for r in runtime.resource_report)
    else:
        assert runtime.resource_report["spill_bytes"] == 0
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
    if dag:
        for mutation in range(6):
            value = (torch.randint(-4, 5, (1, 128), device="cuda").float()/16).bfloat16()
            first_weight = (torch.randint(-4, 5, (128, 128), device="cuda").float()/16).bfloat16()
            second_weight = torch.zeros_like(first_weight) if mutation == 5 else first_weight.flip((0,)).contiguous()
            stream.wait_stream(torch.cuda.current_stream())
            runtime.run_into(output, value, first_weight, second_weight, stream=stream.cuda_stream)
            torch.cuda.synchronize()
            first_rhs = first_weight.T if transpose_b else first_weight
            second_rhs = second_weight.T if transpose_b else second_weight
            gold = ((value.double() @ first_rhs.double()).bfloat16().double() @ second_rhs.double()).bfloat16()
            assert torch.equal(output, gold)
        with torch.cuda.graph(capture := torch.cuda.CUDAGraph(), stream=stream):
            runtime.run_into(output, value, first_weight, second_weight, stream=stream.cuda_stream)
        for _ in range(3):
            capture.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, gold)
    runtime.close()
