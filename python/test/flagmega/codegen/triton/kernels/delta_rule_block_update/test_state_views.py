# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.ops.nn.delta_rule_block_update import delta_rule_block_update
from triton.flagmega.ir.ops.tensors.pack import pack_physical
from triton.flagmega.ir.ops.tensors.unpack import unpack_physical
from triton.flagmega.runtime import load


def _reusable_state_graph(head_axes, private_result=False):
    placement = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    head = fm.SBP.split_contiguous(head_axes) if head_axes else b
    qtype = fm.tensor_type("bfloat16", (1, 2, 4))
    vtype = fm.tensor_type("bfloat16", (1, 4, 4))
    ctypes = fm.tensor_type("bfloat16", (1, 4, 8, 8))
    ptype = fm.tensor_type("float32", (1, 4, 8))
    state_type = fm.RefType("cache", (
        ("untouched", fm.tensor_type("float32", (3, 1))),
        ("matrix", fm.tensor_type(fm.VectorType(fm.DType.FLOAT32, (2, 2)), (3, 4, 2, 2))),
    ))
    specifications = tuple(
        zip(("query", "key", "value", "coefficients", "prefix", "state", "layer"),
            (qtype, qtype, vtype, ctypes, ptype, state_type, fm.tensor_type("int32", ()))))

    class Graph(fm.Module):

        def forward(self):
            parameters = tuple(self.input(name, value, id=name) for name, value in specifications)
            local = tuple(
                fm.F.distributed.force_boxing(
                    node, fm.DistributedType(node.type, (b, head, *(b
                                                                    for _ in range(node.type.rank - 2))), placement))
                for node in parameters[:5])
            view = fm.F.tir.ref_slice(parameters[5], parameters[6], name="view")
            updated = fm.F.nn.delta_rule_block_update(
                *local, view, scale=1., state_field="matrix", state_layout=("layer", "head", "value", "key"),
                state_vector_axes=("value", "key"), name="update",
                metadata={"bufferization.memory_space": "block_local_data"} if private_result else None)
            output = fm.F.tensors.get_item(updated, 0)
            # Each owner consumes a disjoint part of its replica. Publishing
            # only the state owner's private output cannot satisfy this ABI.
            output = fm.F.distributed.sharded_view(
                output, fm.DistributedType(vtype, (b, fm.SBP.split_contiguous((0, 1)), b), placement))
            output = fm.F.math.add(output, output)
            output = fm.F.distributed.force_boxing(output, vtype)
            self.function("worker", parameters, (output, ), attrs={"noinline": True, "reusable": True})
            entry = tuple(self.input("entry_" + name, value, id="entry_" + name) for name, value in specifications)
            first = fm.F.builtin.call(*entry, result_type=vtype, callee="worker",
                                      effect=fm.effect("read_write", "delta_rule_state"), name="first")
            second = fm.F.builtin.call(*entry[:2], first, *entry[3:], result_type=vtype, callee="worker",
                                       effect=fm.effect("read_write", "delta_rule_state"), name="second")
            self.function("main", entry, (second, entry[5]))

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}}).build()


@pytest.mark.parametrize("head_axes", ((), (0, )))
def test_reusable_block_updates_only_selected_packed_layer_and_preserves_output_ownership(tmp_path, head_axes):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA required")
    original = _reusable_state_graph(head_axes)
    namespace = {}
    exec(fm.module_source(original), namespace)
    compiled = Compiler().compile(namespace["MODULE"]).module
    plan = fm.verify_buffer_plan(compiled)
    bindings = dict(plan.function_map["worker"].values)
    for parent_id, view_id in zip(bindings["state"], bindings["view"]):
        parent, view = plan.buffer_map[parent_id], plan.buffer_map[view_id]
        assert view.physical_id == parent.physical_id
        assert view.mem_span.size * 3 == parent.mem_span.size
        assert view.mem_span.is_within(parent.mem_span)
    output_buffer = plan.buffer_map[bindings["update"][0]]
    assert output_buffer.distributed_storage_kind == fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    artifact = write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90", emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert source.count("def _flagmega_function_worker__consumer(") == 1
    assert source.count("    _flagmega_function_worker__consumer(") == 2
    runtime = load(artifact, device="cuda:0")
    query = torch.full((1, 2, 4), 0.5, dtype=torch.bfloat16, device="cuda")
    value = torch.arange(16, device="cuda").reshape(1, 4, 4).bfloat16() / 16
    coefficients = torch.zeros((1, 4, 8, 8), dtype=torch.bfloat16, device="cuda")
    coefficients[:, :, 0, 0] = 0.5
    prefix = torch.zeros((1, 4, 8), device="cuda")
    initial = (torch.arange(192).reshape(3, 4, 4, 4) % 8).float() / 32
    matrix = pack_physical(initial, 4, (2, 2), (2, 3)).cuda()
    untouched = torch.full((3, 1), 23., device="cuda")
    output = torch.empty_like(value)
    expected_state = initial.clone()
    state_buffers = dict(runtime.buffer_plan.function_map["main"].parameters)["entry_state"]
    fields = dict(zip(state_buffers, (untouched, matrix)))

    def arguments(layer):
        inputs = {
            "entry_query": query, "entry_key": query, "entry_value": value, "entry_coefficients": coefficients,
            "entry_prefix": prefix, "entry_layer": layer
        }
        return tuple(output if spec["role"] == "result" else fields[spec["buffer"]] if spec["value"] ==
                     "entry_state" else inputs[spec["value"]] for spec in runtime.external_arguments)

    runtime.prepare(*arguments(2))
    for layer in (2, 0, 1, 2):
        value.add_(0.125)
        expected = value.cpu()
        for _ in range(2):
            expected, final = delta_rule_block_update(query.cpu(), query.cpu(), expected, coefficients.cpu(),
                                                      prefix.cpu(), expected_state[layer], scale=1., torch=torch)
            expected_state[layer] = final
            expected *= 2
        output.fill_(float("nan"))
        runtime.run_into(*arguments(layer))
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(unpack_physical(matrix.cpu(), 4, (2, 2), (2, 3)), expected_state, rtol=0, atol=0)
        assert torch.equal(untouched, torch.full_like(untouched, 23.))
    assert runtime.prepare_count == 1
    assert runtime.resource_report["spill_bytes"] == 0


def test_whole_tuple_placement_cannot_relocate_forwarded_external_state():
    # A node-wide placement applies to every leaf, not just its first tensor.
    # The returned Ref owns no new allocation and must remain caller-owned.
    with pytest.raises(IRVerificationError, match="view.untouched.*external"):
        Compiler().compile(_reusable_state_graph((), private_result=True))
