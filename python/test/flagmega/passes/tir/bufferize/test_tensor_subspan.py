# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions


def slice_graph(*, shape=(3, 16), starts=(1,), ends=(2,), axes=(0,), vector=False, distributed=False,
                producer=False, exported=False, crop=False, split=False, chip=False):
    dtype = fm.vector_type("float32", (2, 2)) if vector else "float32"
    value_type = fm.tensor_type(dtype, shape)
    mesh = fm.Placement((2, 2), "yx", "bb")
    if distributed:
        policies = (fm.SBP.broadcast(),) * len(shape)
        if split:
            policies = (*policies[:-1], fm.SBP.split_contiguous((0, 1)))
        value_type = fm.DistributedType(value_type, policies, mesh)

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", value_type, id="x")
            def add(value, name):
                return (fm.F.math.vectorized_binary(value, value, binary_op="add", name=name, metadata={
                    "selected_vectorization": "test.subspan", "selected_vector_axes": (len(shape) - 1,) * 2,
                    "selected_vector_lanes": (2, 2),
                }) if vector else
                        fm.F.math.add(value, value, name=name))
            value = add(x, "parent") if producer else x
            view = (fm.F.tensors.slice_to_shape(value, shape=ends, name="view") if crop else
                    fm.F.tensors.slice(value, starts=starts, ends=ends, axes=axes, name="view"))
            output = view if exported else add(view, "output")
            self.function("main", (x,), (output,))

    module = Graph(dialect="ntt", stage="frozen_constants", entry="main",
                   metadata={"auto_distribution": {"placement": mesh.to_data()}} if distributed else {}).build()
    if chip:
        module = replace(module, nodes=tuple(replace(node, metadata={**node.metadata,
                          "bufferization.memory_space": "workspace"}) if node.id == "parent" else node
                          for node in module.nodes))
    return module


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("vector", [False, True])
def test_contiguous_slice_is_a_typed_subspan(distributed, vector, tmp_path):
    source = slice_graph(distributed=distributed, vector=vector)
    result = Compiler(CompileOptions(bufferize_opt_level="fast")).compile(source).module
    assert result.node_map["view"].op == "tir.buffer_subspan"
    plan = fm.verify_buffer_plan(result)
    bindings = dict(plan.function_map["main"].values)
    parent = plan.buffer_map[bindings["x"][0]]
    view = plan.buffer_map[bindings["view"][0]]
    assert view.physical_id == parent.physical_id
    assert view.byte_offset == parent.byte_offset + 16 * parent.dtype.itemsize
    assert view.nbytes == 16 * parent.dtype.itemsize
    assert view.mem_span.is_within(parent.mem_span)
    assert fm.load_module(fm.emit_module(result, tmp_path / "result.py")) == result


def test_prefix_crop_is_not_a_copy():
    module = slice_graph(shape=(1, 8), ends=(1, 1), crop=True, distributed=True)
    result = Compiler().compile(module).module
    assert result.node_map["view"].op == "tir.buffer_subspan"


def test_subspan_representation_is_fixed_before_microkernel_proposal():
    result = Compiler().compile(slice_graph(), stop_after="propose-microkernels").module
    assert result.node_map["view"].op == "tir.buffer_subspan"
    from triton.flagmega.passes.pipeline import PIPELINE_GROUPS
    group = next(group for group in PIPELINE_GROUPS if group.name == "TIRPass")
    stages = [item.stage for item in group.passes]
    assert stages.index("lower-tensor-subspans") < stages.index("propose-microkernels")


def test_noncontiguous_columns_still_require_materialization():
    module = slice_graph(shape=(3, 16), starts=(2,), ends=(6,), axes=(1,))
    result = Compiler().compile(module).module
    assert result.node_map["view"].op == "tir.call"


def test_exported_slice_materializes_the_result_abi():
    result = Compiler().compile(slice_graph(exported=True)).module
    assert result.node_map["view"].op == "tir.call"


@pytest.mark.parametrize("producer,split,chip", [(False, False, False), (True, False, False), (True, True, True)])
def test_subspan_device_execution(producer, split, chip, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load

    module = slice_graph(producer=producer, distributed=True, vector=True, split=split, chip=chip)
    result = Compiler().compile(module).module
    assert result.node_map["view"].op == "tir.buffer_subspan"
    if chip:
        plan = fm.verify_buffer_plan(result)
        parent, view = plan.buffer_map["parent"], plan.buffer_map["view"]
        assert view.component_stride_bytes == parent.nbytes == 3 * 4 * 16
        assert view.nbytes == 4 * 16
        assert view.physical_access_span.nbytes == parent.nbytes * 3 + view.nbytes
    artifact = write_artifact(result, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    x = torch.arange(3 * 16 * 4, dtype=torch.float32, device="cuda").reshape(3, 16, 2, 2)
    runtime.prepare(x)
    for _ in range(3):
        torch.testing.assert_close(runtime.run(x), x[1:2] * (4 if producer else 2), rtol=0, atol=0)


def test_owner_stride_cannot_be_changed_in_an_edited_alias():
    result = Compiler().compile(slice_graph(producer=True, distributed=True, split=True, chip=True)).module
    plan = fm.verify_buffer_plan(result)
    view = plan.buffer_map["view"]
    corrupted = replace(plan, buffers=tuple(replace(buffer, owner_stride_bytes=view.nbytes)
                                           if buffer.id == view.id else buffer for buffer in plan.buffers))
    from triton.flagmega.errors import IRVerificationError
    with pytest.raises(IRVerificationError, match="owner stride"):
        fm.verify_buffer_plan(replace(result, metadata={**result.metadata, "buffer_plan": corrupted.to_data()}))


def test_subspan_argument_keeps_the_caller_base_offset(tmp_path):
    class Graph(fm.Module):
        def forward(self):
            p = self.input("p", fm.tensor_type("float32", (1, 16)), id="p")
            output = fm.F.math.add(p, p, name="worker_output")
            self.function("worker", (p,), (output,), attrs={"reusable": True, "noinline": True})
            x = self.input("x", fm.tensor_type("float32", (3, 16)), id="x")
            view = fm.F.tensors.slice(x, starts=(1,), ends=(2,), axes=(0,), name="view")
            call = fm.F.builtin.call(view, callee="worker", result_type=output.type, name="call")
            self.function("main", (x,), (call,))
    result = Compiler().compile(Graph(dialect="ntt", stage="frozen_constants", entry="main").build()).module
    plan = fm.verify_buffer_plan(result)
    assert result.node_map["view"].op == "tir.buffer_subspan"
    assert plan.buffer_map["view"].byte_offset == 64
    from triton.flagmega.codegen.triton import render_triton_package
    render_triton_package(result, tmp_path)
    assert "+ 16" in (tmp_path / "generated_kernels.py").read_text()


def test_shard_backing_promotion_rebases_subspan_coordinates():
    source = slice_graph(producer=True, distributed=True, split=True, chip=True)
    view_type = source.node_map["view"].type
    target = replace(view_type, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((1,))))
    retarget = fm.Node("retarget", "distributed.sharded_view", ("view",), target, attrs={"new_type": target})
    source = replace(source, nodes=(*source.nodes[:-1], retarget,
                                   replace(source.nodes[-1], inputs=("retarget", "retarget"), type=target)))
    result = Compiler().compile(source).module
    plan = fm.verify_buffer_plan(result)
    assert plan.buffer_map["parent"].distributed_storage_kind.value == "canonical_global"
    assert plan.buffer_map["view"].byte_offset == 16 * 4
    assert plan.buffer_map["view"].owner_stride_bytes is None
    assert plan.buffer_map["parent"].mem_span.buffer.live_end >= plan.buffer_map["view"].live_end


@pytest.mark.parametrize("alignment,expected", [(16, "tir.buffer_subspan"), (128, "tir.call")])
def test_subspan_selection_obeys_the_declared_consumer_abi(alignment, expected):
    from triton.flagmega.passes.tir.lower_tensor_subspans import lower_tensor_subspans
    planned = Compiler().compile(slice_graph(), stop_after="plan-tir-alignments").module
    callee = planned.node_map["output"].attrs["callee"]
    planned = replace(planned, kernel_definitions=tuple(
        replace(kernel, parameters=tuple(replace(p, alignment_bytes=alignment) if p.name in {"lhs", "rhs"} else p
                                         for p in kernel.parameters)) if kernel.name == callee else kernel
        for kernel in planned.kernel_definitions))
    result = lower_tensor_subspans(planned)
    assert result.node_map["view"].op == expected


def test_real_transfer_alignment_propagates_through_a_subspan():
    from python.test.flagmega.passes.tir.bufferize.test_shared_workspace import _selected_module
    from triton.flagmega.passes.tir.bufferize.alignment import transfer_source_alignment_requirements
    source = _selected_module(source_alignment=128)
    weight = source.node_map["fused_weight"]
    parent = replace(weight, id="parent", type=fm.tensor_type("bfloat16", (2049, 4096)))
    view = fm.Node(weight.id, "tir.buffer_subspan", (parent.id,), weight.type,
                   attrs={"offsets": (1, 0), "shape": (2048, 4096)})
    source = fm.verify_module(replace(source, nodes=tuple(
        replacement for node in source.nodes for replacement in ((parent, view) if node.id == weight.id else (node,)))))
    requirements = transfer_source_alignment_requirements(source)
    assert requirements[view.id] == requirements[parent.id] == 128


def test_declared_alignment_applies_to_every_owner_component(tmp_path):
    source = slice_graph(producer=True, distributed=True, split=True, chip=True, starts=(0,), ends=(1,))
    compiler = Compiler()
    planned = compiler.compile(source, stop_after="plan-tir-alignments").module
    callee = planned.node_map["output"].attrs["callee"]
    planned = replace(planned, kernel_definitions=tuple(
        replace(kernel, parameters=tuple(replace(p, alignment_bytes=128) if p.name in {"lhs", "rhs"} else p
                                         for p in kernel.parameters)) if kernel.name == callee else kernel
        for kernel in planned.kernel_definitions))
    result = compiler.compile(planned).module
    plan = fm.verify_buffer_plan(result)
    assert result.node_map["view"].op == "tir.buffer_subspan"
    parent, view = plan.buffer_map["parent"], plan.buffer_map["view"]
    assert parent.nbytes == 48
    assert parent.component_stride_bytes == view.component_stride_bytes == 128
    assert view.byte_offset % 128 == 0
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required for padded-owner execution")
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    runtime = load(write_artifact(result, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True),
                   device="cuda:0")
    x = torch.arange(48, dtype=torch.float32, device="cuda").reshape(3, 16)
    runtime.prepare(x)
    torch.testing.assert_close(runtime.run(x), x[:1] * 4, rtol=0, atol=0)
