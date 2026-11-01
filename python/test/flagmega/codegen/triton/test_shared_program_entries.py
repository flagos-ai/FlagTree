# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Repeated physical calls share code while keeping their actual bindings."""

import pytest
from copy import deepcopy
import math

from triton.flagmega.codegen.triton import render_triton_package
from triton.flagmega.errors import ArtifactError
from triton.flagmega.runtime.kernel_programs import verified_kernel_programs
from python.test.flagmega.codegen.triton.test_function_programs import compile_repeated_stages
from python.test.flagmega.codegen.triton.test_kernel_programs import import_source


@pytest.mark.parametrize("packed", [False, True])
def test_repeated_calls_share_jit_definitions_with_distinct_actuals(tmp_path, packed):
    module = compile_repeated_stages(packed=packed)
    package = render_triton_package(module, tmp_path)
    source = import_source(tmp_path / "generated_kernels.py")
    kernels = [getattr(source, spec["symbol"]) for spec in package["kernels"]]
    assert len({id(kernel) for kernel in kernels}) == 6
    for stage in range(6):
        assert kernels[stage] is kernels[stage + 6]
    assert package["kernels"][0]["argument_bindings"] != package["kernels"][6]["argument_bindings"]
    assert len(verified_kernel_programs(package, kernels, module)) == 12


@pytest.mark.parametrize("change", ["offset", "source", "record", "scope", "definition"])
def test_shared_entry_rejects_forged_manifest_and_source_binding(tmp_path, change):
    if change == "record":
        from python.test.flagmega.codegen.triton.pipelines.conftest import _compile_pipeline_module
        module = _compile_pipeline_module(reusable=True, worker_depth=4, program_dag=True,
                                         implementation="tir.dense_matmul.tensor_descriptor_gemv")
    else:
        module = compile_repeated_stages()
    package = render_triton_package(module, tmp_path)
    source = import_source(tmp_path / "generated_kernels.py")
    kernels = [getattr(source, spec["symbol"]) for spec in package["kernels"]]
    assert len(verified_kernel_programs(package, kernels, module)) == len(kernels)
    forged = deepcopy(package)
    spec = forged["kernels"][-1] if change == "record" else forged["kernels"][6]
    if change == "record":
        from triton.flagmega.codegen.triton.function_pool_abi import CALL_RECORD

        binding = spec["argument_bindings"][kernels[-1].arg_names.index(CALL_RECORD)]
        assert binding["kind"] == "immediate" and binding["dtype"] == "int64"
        binding["value"] += 1
    elif change == "definition":
        spec["entry_function"] = "worker.join"
    else:
        binding = next(b for b in spec["argument_bindings"] if b["kind"] == "pointer")
        if change == "source":
            binding["source"] = 0
        elif change == "scope":
            binding["scope_stride_bytes"] += 16
        else:
            binding["byte_offset"] += 2
    source.FLAGMEGA_KERNEL_PROGRAMS = forged["kernels"]
    with pytest.raises(ArtifactError, match="typed physical call ABI"):
        verified_kernel_programs(forged, kernels, module)


def test_pointer_projection_is_a_byte_addressed_no_copy_view():
    torch = pytest.importorskip("torch")
    from triton.flagmega.runtime.program_arguments import project_arguments

    root = torch.arange(32, dtype=torch.int32)
    binding = dict(kind="pointer", source=0, dtype="int32", byte_offset=12, scope_stride_bytes=64)
    (view,) = project_arguments((binding,), (root,))
    assert view.data_ptr() == root.data_ptr() + 12
    view[0] = -33
    assert root[3] == -33
    assert project_arguments((dict(kind="immediate", dtype="int64", value=2**45),), ()) == (2**45,)


@pytest.mark.parametrize("value,dtype", [(True, "bool"), (-0., "float64")])
def test_identity_scalar_projection_preserves_boolean_and_signed_zero(value, dtype):
    from triton.flagmega.runtime.program_arguments import project_arguments

    (actual,) = project_arguments((dict(kind="scalar_argument", source=0, dtype=dtype, add=0),), (value,))
    assert type(actual) is type(value)
    if dtype == "float64":
        assert math.copysign(1., actual) == -1.


def test_shared_descriptor_entry_preserves_each_call_resource(tmp_path):
    from python.test.flagmega.codegen.triton.descriptors.conftest import _two_call_descriptor_module

    module = _two_call_descriptor_module(program_dag=True)
    package = render_triton_package(module, tmp_path)
    source = import_source(tmp_path / "generated_kernels.py")
    assert source.program_first is source.program_second
    specs = package["host_tensor_descriptor_specs"]
    assert len(specs) == 2
    indices = [package["signature_arguments"].index(spec["name"]) for spec in specs]
    resources = [[binding["source"] for binding in launch["argument_bindings"]
                  if binding["kind"] == "argument" and binding["source"] in indices]
                 for launch in package["kernels"] if launch["entry_function"] == "worker"]
    assert resources == [[indices[0]], [indices[1]]]


def test_entry_identity_preserves_the_position_of_device_fences():
    from triton.flagmega.codegen.triton.function_programs import project_function_programs
    from triton.flagmega.codegen.triton.tir_package import describe_tir_package

    module = compile_repeated_stages(packed=True)
    descriptor = describe_tir_package(module)
    programs = [deepcopy(descriptor["programs"][i]) for i in (0, 6)]
    launches = [deepcopy(descriptor["kernels"][i]) for i in (0, 6)]
    fence = dict(kind="barrier", scope="device")
    calls = [next(e for e in p["entry_events"] if e["kind"] == "function_call") for p in programs]
    programs[0]["entry_events"] = [fence, calls[0]]
    programs[1]["entry_events"] = [calls[1], fence]
    for program in programs:
        program.pop("entry_alias", None)
    project_function_programs(module, descriptor, programs, launches)
    assert launches[0]["entry_symbol"] != launches[1]["entry_symbol"]


def test_device_shared_pool_records_select_distinct_weights_and_rebind_inputs(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from python.test.flagmega.codegen.triton.pipelines.conftest import _compile_pipeline_module
    from python.test.flagmega.codegen.triton.pipelines.test_pipeline_runtime import _checkpoint

    weights = tuple((torch.eye(128).roll(index + 1, 0) *
                     (((torch.arange(128) + index) % 3 + 1).float()[:, None] / 4)).bfloat16()
                    for index in range(8))
    checkpoint = {f"{prefix}_weight_{index}": weights[group * 4 + index]
                  for group, prefix in enumerate(("first", "second")) for index in range(4)}
    module = _compile_pipeline_module(reusable=True, worker_depth=4, program_dag=True,
                                     implementation="tir.dense_matmul.tensor_descriptor_gemv")
    runtime = load(write_artifact(module, tmp_path / "artifact", target="nvidia-sm90",
                   checkpoint=_checkpoint(checkpoint), emit_executable=True), device="cuda:0")
    guard = torch.full((160,), -123., dtype=torch.bfloat16, device="cuda")
    output = guard[16:-16].reshape(1, 128)
    value = torch.zeros_like(output)
    stream = torch.cuda.Stream()

    def gold(value):
        for weight in weights:
            value = (value.double() @ weight.to("cuda").double().T).bfloat16()
        return value

    try:
        runtime.prepare(value, output=output)
        assert len(runtime._function_offset_values) == 1
        first, second = [i for i, spec in enumerate(runtime.codegen["kernels"]) if spec["entry_function"] == "worker"]
        assert runtime.kernels[first] is runtime.kernels[second]
        assert runtime._prepared[first].compiled_kernel is runtime._prepared[second].compiled_kernel
        assert all(r["spill_bytes"] == 0 for r in runtime.resource_report)
        for mutation in range(6):
            value = (torch.randint(-8, 9, (1, 128), device="cuda").float() / 16).bfloat16()
            if mutation == 5:
                value.zero_()
            stream.wait_stream(torch.cuda.current_stream())
            runtime.run_into(output, value, stream=stream.cuda_stream)
            torch.cuda.synchronize()
            assert torch.equal(output, gold(value))
            assert bool((guard[:16] == -123).all() & (guard[-16:] == -123).all())
        with torch.cuda.graph(capture := torch.cuda.CUDAGraph(), stream=stream):
            runtime.run_into(output, value, stream=stream.cuda_stream)
        value.copy_(torch.arange(128, device="cuda").reshape(1, 128).bfloat16() / 128)
        for _ in range(3):
            capture.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, gold(value))
        assert bool((guard[:16] == -123).all() & (guard[-16:] == -123).all())
    finally:
        runtime.close()
