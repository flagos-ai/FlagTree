# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm


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
