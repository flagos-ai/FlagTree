"""Immutable call-record snapshots must follow both supported pointer ABIs."""
import copy

import pytest
from triton.flagmega.errors import CodegenError

from .snapshot_offsets import transform_descriptor


def descriptor(expression, *, producer=False):
    definition = dict(
        function="test.consumer", signature_arguments=["_flagmega_call_offsets"],
        function_pool_abi=dict(record_width=4),
        schedule=dict(consumer_events=[dict(kind="tir.kernel_call", arguments=expression),
                                       dict(kind="tir.kernel_call", arguments=expression)],
                      producer_events=[dict(kind="tir.kernel_call", arguments=expression)] if producer else []))
    return dict(device_functions=[definition], programs=[],
                implementation_override=dict(inputs={}))


@pytest.mark.parametrize("expression", [
    "tl.load(_flagmega_call_offsets + 2, volatile=True)",
    "tl.load(_flagmega_function_offsets.to(tl.pointer_type(tl.int64)) + _flagmega_call_offsets + 2)",
])
def test_snapshot_replaces_shared_column_for_both_pointer_abis(expression):
    result = transform_descriptor(descriptor(expression))
    definition = result["device_functions"][0]
    assert definition["immutable_call_record_bindings"] == [
        dict(column=2, name="_flagmega_record_column_2")]
    assert [event["arguments"] for event in definition["schedule"]["consumer_events"]] == [
        "_flagmega_record_column_2", "_flagmega_record_column_2"]


def test_v2_snapshot_rejects_column_outside_verified_record():
    expression = "tl.load(_flagmega_function_offsets.to(tl.pointer_type(tl.int64)) + _flagmega_call_offsets + 4)"
    with pytest.raises(CodegenError, match="Offset outside record width"):
        transform_descriptor(descriptor(expression))


def test_v2_snapshot_rejects_multiple_roles():
    expression = "tl.load(_flagmega_function_offsets.to(tl.pointer_type(tl.int64)) + _flagmega_call_offsets + 2)"
    with pytest.raises(CodegenError, match="one consumer role"):
        transform_descriptor(descriptor(expression, producer=True))


def test_affine_offset_expression_stays_outside_snapshot_rewrite():
    expression = "(4096 + ((tl.cast(_flagmega_call_offsets, tl.int64) - 0) // 4) * 8192)"
    value = descriptor(expression)
    before = copy.deepcopy(value)
    with pytest.raises(CodegenError, match="Expected duplicate immutable offset uses"):
        transform_descriptor(value)
    assert value == before
