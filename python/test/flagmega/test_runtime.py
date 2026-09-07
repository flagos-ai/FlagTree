# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

import triton.flagmega.runtime.prepared as prepared_runtime
from triton.flagmega.artifacts import load_artifact, write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import RuntimeContractError
from triton.flagmega import ir as fm
from triton.flagmega.runtime import PreparedKernel, ResourceContract, load, prepare_jit_kernel


class _Metadata:
    num_warps = 4
    shared = 1024
    ptxas_stack_frame_bytes = 0
    ptxas_spill_store_bytes = 0
    ptxas_spill_load_bytes = 0


class _FakeCompiled:
    metadata = _Metadata()
    n_regs = 32
    n_spills = 0
    function = object()
    packed_metadata = object()
    name = "fake"

    @property
    def run(self):
        return lambda *args: None


def _add_module(*, op: str = "math.add"):
    builder = fm.IRBuilder(dialect="high_level", stage="imported", metadata={"model": "runtime-add"})
    value_type = fm.tensor_type("float32", [257])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call(op, [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")


def test_prepared_kernel_binds_static_and_dynamic_arguments():
    prepared = PreparedKernel(
        _FakeCompiled(),
        ("dynamic-0", "static", "dynamic-1"),
        (0, 2),
        grid=(8,),
        contract=ResourceContract(4, 1),
    )

    assert prepared.dynamic_argument_count == 2
    assert prepared.static_argument_indices == (1,)
    assert prepared.grid == (8, 1, 1)
    with pytest.raises(RuntimeContractError, match="expects 2"):
        prepared.launch("only-one")


def test_prepared_kernel_owns_and_scopes_compiler_global_scratch(monkeypatch):
    from triton.runtime import _allocation

    class ScratchMetadata(_Metadata):
        global_scratch_size = 8
        global_scratch_align = 16
        num_ctas = 1

    class ScratchCompiled(_FakeCompiled):
        metadata = ScratchMetadata()

        def __init__(self):
            self.seen_buffer = None

        def launch_metadata(self, grid, stream, *arguments):
            return None

        @property
        def run(self):
            def launch(*args):
                self.seen_buffer = _allocation._allocator.get()(32, 16, None)

            return launch

    class Scratch:
        nbytes = 32

        def __call__(self, size, alignment, stream):
            assert (size, alignment) == (32, 16)
            return "prepared-scratch"

    scratch = Scratch()
    monkeypatch.setattr(
        prepared_runtime,
        "_prepare_global_scratch",
        lambda compiled, arguments, grid: scratch,
    )
    compiled = ScratchCompiled()
    prepared = PreparedKernel(
        compiled,
        (),
        (),
        grid=(4,),
        contract=ResourceContract(4, 1),
    )
    previous_allocator = lambda size, alignment, stream: "previous"
    token = _allocation._allocator.set(previous_allocator)
    try:
        prepared.launch()
        assert compiled.seen_buffer == "prepared-scratch"
        assert _allocation._allocator.get() is previous_allocator
        assert prepared.resource_report["global_scratch_bytes"] == 32
    finally:
        _allocation._allocator.reset(token)


def test_prepare_rejects_launch_options_outside_resource_contract():
    class Kernel:
        def run(self, *args, **kwargs):
            raise AssertionError("resource contract should reject before compilation")

    with pytest.raises(RuntimeContractError, match="requires 4"):
        prepare_jit_kernel(
            Kernel(),
            (),
            (),
            grid=(1,),
            contract=ResourceContract(4, 1),
            num_warps=8,
        )


def test_prepare_rejects_requested_residency_that_exceeds_register_file():
    class Kernel:
        def run(self, *args, **kwargs):
            return _FakeCompiled()

    with pytest.raises(RuntimeContractError, match="resident block"):
        prepare_jit_kernel(
            Kernel(),
            (),
            (),
            grid=(1,),
            contract=ResourceContract(4, 3, register_file_capacity_per_sm=8192),
            num_warps=4,
        )


def test_prepared_direct_c_launcher_executes_without_second_jit():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the prepared-launch integration test")
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(lhs, rhs, output, size: tl.constexpr):
        offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
        mask = offsets < size
        tl.store(output + offsets, tl.load(lhs + offsets, mask=mask) + tl.load(rhs + offsets, mask=mask), mask=mask)

    lhs = torch.randn((257,), device="cuda", dtype=torch.float32)
    rhs = torch.randn((257,), device="cuda", dtype=torch.float32)
    output = torch.empty_like(lhs)
    prepared = prepare_jit_kernel(
        add_kernel,
        (lhs, rhs, output, lhs.numel()),
        (0, 1, 2),
        grid=(triton.cdiv(lhs.numel(), 128),),
        contract=ResourceContract(1, 1),
        num_warps=1,
    )

    lhs.fill_(2.0)
    rhs.fill_(3.0)
    prepared.launch(lhs, rhs, output)
    torch.cuda.synchronize()

    torch.testing.assert_close(output, torch.full_like(output, 5.0))


def test_executable_artifact_load_prepare_run_into_on_h800(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required for the executable artifact integration test")
    compiled = Compiler().compile(_add_module()).module
    artifact = write_artifact(
        compiled,
        tmp_path / "add-artifact",
        target="nvidia-sm90",
        emit_executable=True,
    )

    manifest, _ = load_artifact(artifact)
    runtime = load(artifact, device="cuda:0")
    lhs = torch.full((257,), 2.0, dtype=torch.float32, device="cuda:0")
    rhs = torch.full((257,), 3.0, dtype=torch.float32, device="cuda:0")
    output = torch.empty_like(lhs)
    runtime.prepare(lhs, rhs, output=output)
    runtime.run_into(output, lhs, rhs)
    torch.cuda.synchronize()

    assert manifest["status"] == "executable"
    assert manifest["codegen"]["kind"] == "tir_call_graph/v1"
    assert runtime.prepare_count == 1
    torch.testing.assert_close(output, torch.full_like(output, 5.0))

    lhs.fill_(7.0)
    runtime.run_into(output, lhs, rhs)
    torch.cuda.synchronize()
    assert runtime.prepare_count == 1
    torch.testing.assert_close(output, torch.full_like(output, 10.0))


def test_executable_codegen_uses_selected_mul_kernel_template(tmp_path):
    compiled = Compiler().compile(_add_module(op="math.mul")).module
    artifact = write_artifact(
        compiled,
        tmp_path / "mul-artifact",
        target="nvidia-sm90",
        emit_executable=True,
    )
    manifest, _ = load_artifact(artifact)
    source = (artifact / manifest["codegen"]["source"]).read_text("utf-8")

    assert manifest["codegen"]["kind"] == "tir_call_graph/v1"
    call = next(
        value
        for value in manifest["codegen"]["runtime_binding"]["call_abi"]["kernel_calls"]
        if value["semantic_op"] == "math.mul"
    )
    assert call["variant"] == "mul"
    assert {tuple(value.items()) for value in manifest["codegen"]["kernel_template_specs"]} == {
        (("kernel", "distributed_boxing"), ("variant", "tensor_load")),
        (("kernel", "elementwise"), ("variant", "mul")),
    }
    assert "# flagmega-kernel: elementwise/mul platform=generic" in source
    assert "_flagmega_elementwise_mul(" in source
