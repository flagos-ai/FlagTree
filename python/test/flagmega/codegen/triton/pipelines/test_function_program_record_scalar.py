"""Split GPU entries must retain the scalar call-record ABI."""

from types import SimpleNamespace

from triton.flagmega.codegen.triton.function_pool_abi import CALL_RECORD, OFFSET_TABLE
from triton.flagmega.codegen.triton.function_programs import _actual_bindings


def test_projected_function_program_binds_a_scalar_record_displacement():
    definition = {"function_pool_abi": {"pools": [], "record_width": 1,
        "calls": [{"caller": "main", "call": "layer", "record_offset": 5}]}}
    actuals = _actual_bindings(SimpleNamespace(entry="main"), {"pools": []}, definition,
        {"call": "layer", "descriptor_argument_bindings": {}}, [OFFSET_TABLE, CALL_RECORD],
        {"arguments": [], "pools": []}, {"arguments": [], "results": [], "memory_pools": []})
    assert actuals[CALL_RECORD] == {"kind": "immediate", "dtype": "int64", "value": 5}
    assert actuals[OFFSET_TABLE] == {"kind": "argument", "source": OFFSET_TABLE}
