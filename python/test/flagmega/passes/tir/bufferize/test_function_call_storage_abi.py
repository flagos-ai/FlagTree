# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm


@pytest.mark.parametrize("materialize", [False, True])
def test_chip_boundary_view_keeps_canonical_storage_in_the_callee(materialize):
    placement = fm.Placement((4, 2, 2), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (16, 8))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement)
    split = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()), placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            parameter = self.input("parameter", split, id="parameter")
            value = self.input("value", broadcast, id="value")
            source = fm.F.math.silu(value, name="produced") if materialize else value
            viewed = fm.F.distributed.sharded_view(source, split, name="viewed")
            call = fm.F.builtin.call(
                viewed, result_type=split, callee="worker", name="call",
            )
            self.function("main", (value,), (call,))
            self.function("worker", (parameter,), (parameter,),
                          attrs={"reusable": True, "noinline": True})

    module = Graph().build()
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    plan = fm.make_buffer_plan(module)
    formal, actual = (plan.buffer_map[name] for name in ("parameter", "viewed"))
    assert formal.distributed_type == actual.distributed_type == split
    assert formal.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert actual.distributed_storage_kind is formal.distributed_storage_kind
    assert actual.nbytes == formal.nbytes == 16 * 8 * 4
    assert actual.strides == formal.strides == (8, 1)
    assert actual.mem_span.must_alias(plan.buffer_map["produced" if materialize else "value"].mem_span)
    assert dict(plan.call_map["call"].arguments)[formal.id] == actual.id


@pytest.mark.parametrize("partial_second", [False, True])
def test_canonical_call_requirement_does_not_promote_an_unrelated_tuple_field(partial_second):
    placement = fm.Placement((4, 2, 2), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (16, 8))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement)
    split = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()), placement,
    )
    second_type = (fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement,
                                     partial=fm.SBP.partial((0,))) if partial_second else split)
    pair_type = fm.TupleType((split, second_type))

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            parameter = self.input("parameter", pair_type, id="parameter")
            full = self.input("full", broadcast, id="full")
            local = self.input("local", second_type, id="local")
            viewed = fm.F.distributed.sharded_view(full, split, name="viewed")
            pair = fm.F.builtin.tuple(viewed, local, name="pair")
            call = fm.F.builtin.call(pair, result_type=pair_type, callee="worker", name="call")
            result = fm.F.tensors.get_item(call, 0, name="result")
            self.function("main", (full, local), (result,))
            self.function("worker", (parameter,), (parameter,),
                          attrs={"reusable": True, "noinline": True})

    module = Graph().build()
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    plan = fm.make_buffer_plan(module)
    first, second = (plan.buffer_map[f"parameter.{i}"] for i in range(2))
    assert first.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert first.nbytes == 16 * 8 * 4
    expected = (fm.DistributedBufferStorageKind.COMPACT_PER_OWNER if partial_second
                else fm.DistributedBufferStorageKind.COMPACT_PER_CHIP)
    assert second.distributed_storage_kind is expected
    assert second.nbytes == (16 if partial_second else 4) * 8 * 4
    assert plan.buffer_map["local"].distributed_storage_kind is second.distributed_storage_kind


@pytest.mark.parametrize("view_first", [False, True])
def test_all_call_sites_agree_before_either_function_is_allocated(view_first):
    placement = fm.Placement((4, 2, 2), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (16, 8))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement)
    split = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()), placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            parameter = self.input("parameter", split, id="parameter")
            full = self.input("full", broadcast, id="full")
            local = self.input("local", split, id="local")
            viewed = fm.F.distributed.sharded_view(full, split, name="viewed")
            actuals = (viewed, local) if view_first else (local, viewed)
            calls = [fm.F.builtin.call(value, result_type=split, callee="worker", name=f"call{i}")
                     for i, value in enumerate(actuals)]
            self.function("main", (full, local), tuple(calls))
            self.function("worker", (parameter,), (parameter,),
                          attrs={"reusable": True, "noinline": True})

    module = Graph().build()
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    plan = fm.make_buffer_plan(module)
    formal = plan.buffer_map["parameter"]
    for call in plan.call_map.values():
        actual = plan.buffer_map[dict(call.arguments)[formal.id]]
        assert actual.distributed_storage_kind is formal.distributed_storage_kind
        assert actual.strides == formal.strides
        assert actual.nbytes == formal.nbytes == 16 * 8 * 4


def test_loop_backedge_retains_the_canonical_carried_coordinate_contract():
    from triton.flagmega.ir.ops.core import construction_scope

    placement = fm.Placement((4, 2, 2), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (16, 8))
    index_type = fm.tensor_type("int64", ())
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(),) * 2, placement)
    split = fm.DistributedType(
        tensor, (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()), placement,
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    iv = builder.var("iv", index_type, id="iv")
    carry = builder.var("carry", split, id="carry")
    next_value = builder.call("tir.kernel", (carry,), split, id="next_value")
    body = builder.region("body", (iv, carry), (next_value,), iter_count=1)
    full = builder.var("full", broadcast, id="full")
    with construction_scope(builder):
        viewed = fm.F.distributed.sharded_view(full, split, name="viewed")
        lower = fm.F.builtin.scalar_const(index_type, 0, name="lower")
        upper = fm.F.builtin.scalar_const(index_type, 3, name="upper")
        step = fm.F.builtin.scalar_const(index_type, 1, name="step")
        loop = fm.F.control.for_loop(lower, upper, step, viewed, body=body, name="loop")
        output = fm.F.tensors.get_item(loop, 0, name="output")
    builder.function("main", (full,), (output,))
    module = builder.build(entry="main")
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    plan = fm.make_buffer_plan(module)
    for name in ("carry", "next_value"):
        buffer = plan.buffer_map[name]
        assert buffer.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
        assert buffer.strides == (8, 1)
        assert buffer.nbytes == 16 * 8 * 4


def test_materialized_function_argument_uses_callee_canonical_storage_abi():
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("float32", (4, 16))
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            parameter = self.input("parameter", split, id="parameter")
            restored = fm.F.distributed.sharded_view(
                parameter, broadcast, name="restored")
            value = self.input("value", broadcast, id="value")
            boxed = fm.F.distributed.boxing(value, split, name="boxed")
            call = fm.F.builtin.call(
                boxed, result_type=broadcast, callee="worker", name="call")
            self.function("main", (value,), (call,))
            self.function(
                "worker",
                (parameter,),
                (restored,),
                attrs={"reusable": True, "noinline": True},
            )

    plan = fm.make_buffer_plan(Graph().build())
    formal = plan.buffer_map["parameter"]
    actual = plan.buffer_map["boxed"]

    assert formal.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert actual.distributed_storage_kind is formal.distributed_storage_kind
    assert actual.distributed_type == formal.distributed_type
    assert actual.strides == formal.strides
    assert actual.nbytes == formal.nbytes == 4 * 16 * 4
    call = plan.call_map["call"]
    assert dict(call.arguments)[formal.id] == actual.id
    result = plan.buffer_map[dict(call.results)["restored"]]
    assert result.distributed_type == broadcast
    assert result.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert result.mem_span.must_alias(actual.mem_span)


def test_chip_boundary_promotes_when_callee_view_changes_block_owners():
    placement = fm.Placement((4, 8, 16), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (16, 8, 1))
    broadcast = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    chip_split = fm.DistributedType(
        tensor,
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    block_split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,)), fm.SBP.broadcast()),
        placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            parameter = self.input("parameter", chip_split, id="parameter")
            viewed = fm.F.distributed.sharded_view(
                parameter, block_split, name="viewed")
            value = self.input("value", broadcast, id="value")
            boxed = fm.F.distributed.boxing(value, chip_split, name="boxed")
            call = fm.F.builtin.call(
                boxed, result_type=block_split, callee="worker", name="call")
            self.function("main", (value,), (call,))
            self.function(
                "worker",
                (parameter,),
                (viewed,),
                attrs={"reusable": True, "noinline": True},
            )

    module = Graph().build()
    module = replace(module, metadata={"launch_contract": {"grid_mesh": placement.to_data()}})
    plan = fm.make_buffer_plan(module)
    formal = plan.buffer_map["parameter"]
    actual = plan.buffer_map["boxed"]
    assert formal.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert actual.distributed_storage_kind is formal.distributed_storage_kind
    assert actual.nbytes == formal.nbytes == 16 * 8 * 1 * 4


def test_tuple_projected_function_argument_propagates_canonical_storage_to_field():
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("float32", (4, 16))
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            producer_parameter = self.input(
                "producer_parameter", split, id="producer_parameter")
            produced = fm.F.math.silu(producer_parameter, name="produced")

            worker_parameter = self.input(
                "worker_parameter", split, id="worker_parameter")

            value = self.input("value", split, id="value")
            producer_call = fm.F.builtin.call(
                value,
                result_type=fm.TupleType((split,)),
                callee="producer",
                name="producer_call",
            )
            projected = fm.F.tensors.get_item(
                producer_call, 0, name="projected")
            worker_call = fm.F.builtin.call(
                projected,
                result_type=split,
                callee="worker",
                name="worker_call",
            )
            self.function("main", (value,), (worker_call,))
            self.function(
                "producer",
                (producer_parameter,),
                (produced,),
                attrs={"reusable": True, "noinline": True},
            )
            self.function(
                "worker",
                (worker_parameter,),
                (worker_parameter,),
                attrs={"reusable": True, "noinline": True},
            )

    plan = fm.make_buffer_plan(Graph().build())
    producer_result = plan.buffer_map[dict(plan.call_map["producer_call"].results)["produced"]]
    projected = plan.buffer_map[dict(plan.function_map["main"].values)["projected"][0]]
    worker_formal = plan.buffer_map["worker_parameter"]

    assert producer_result.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert projected.id == producer_result.id
    assert projected.distributed_storage_kind is worker_formal.distributed_storage_kind
    assert projected.shape == worker_formal.shape
    assert projected.strides == worker_formal.strides


def test_tuple_assembly_propagates_function_argument_abi_to_each_operand():
    placement = fm.Placement((2, 2), "yx", "bb")
    tensor = fm.tensor_type("float32", (4, 16))
    split = fm.DistributedType(
        tensor,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
        placement,
    )
    pair_type = fm.TupleType((split, split))

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="tir", stage="selected_tir", entry="main")

        def forward(self):
            pair_parameter = self.input(
                "pair_parameter", pair_type, id="pair_parameter")

            value = self.input("value", split, id="value")
            first = fm.F.math.silu(value, name="first")
            second = fm.F.math.silu(value, name="second")
            pair = fm.F.builtin.tuple(first, second, name="pair")
            call = fm.F.builtin.call(
                pair,
                result_type=pair_type,
                callee="worker",
                name="call",
            )
            result = fm.F.tensors.get_item(call, 0, name="result")
            self.function("main", (value,), (result,))
            self.function(
                "worker",
                (pair_parameter,),
                (pair_parameter,),
                attrs={"reusable": True, "noinline": True},
            )

    plan = fm.make_buffer_plan(Graph().build())
    values = dict(plan.function_map["main"].values)
    formal_ids = plan.function_map["worker"].parameters[0][1]

    assert values["pair"] == (values["first"][0], values["second"][0])
    for actual_id, formal_id in zip(values["pair"], formal_ids, strict=True):
        actual = plan.buffer_map[actual_id]
        formal = plan.buffer_map[formal_id]
        assert actual.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
        assert actual.distributed_storage_kind is formal.distributed_storage_kind
        assert actual.shape == formal.shape
        assert actual.strides == formal.strides
