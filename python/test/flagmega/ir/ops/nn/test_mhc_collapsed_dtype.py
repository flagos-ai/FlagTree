# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The collapse's storage precision is independent of its BF16 input streams."""

import subprocess
import sys

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.mhc_pre import MhcPre

from python.test.flagmega.ir.ops.nn.test_mhc import _pre_attrs, _pre_types


class WideCollapse(fm.Module):
    def __init__(self, dtype=None, lanes=()):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.dtype, self.lanes = dtype, lanes

    def forward(self):
        types = {"streams": fm.tensor_type("bfloat16", (1, 4, 16)),
                 "projection": fm.tensor_type("float32", (24, 64)),
                 "base": fm.tensor_type("float32", (24,)), "scale": fm.tensor_type("float32", (3,))}
        values = tuple(self.input(name, value, id=name) for name, value in types.items())
        pre = fm.F.nn.mhc_pre(*values, **_pre_attrs(), collapsed_dtype=self.dtype,
                              output_lanes=self.lanes, name="pre")
        self.function("main", values, fm.F.tensors.get_items(pre, 0, 1, 2, name_prefix="field"))


def rounding_inputs():
    streams = torch.zeros((1, 4, 16), dtype=torch.bfloat16)
    streams[:, 0] = torch.linspace(.25, 4., 16).bfloat16()
    return {"streams": streams, "projection": torch.zeros((24, 64)),
            "base": torch.zeros(24), "scale": torch.zeros(3)}


def test_unrounded_collapse_has_an_independent_float32_result_type():
    result = MhcPre.infer_type(_pre_types(), _pre_attrs(collapsed_dtype="float32"))
    assert result.fields[2].dtype == fm.DType.FLOAT32
    assert result.fields[:2] == MhcPre.infer_type(_pre_types(), _pre_attrs()).fields[:2]


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
def test_wide_collapse_retains_precision_before_storage_rounding(lanes):
    inputs = rounding_inputs()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    legacy = evaluator.run(WideCollapse().build(), inputs)
    wide = evaluator.run(WideCollapse("float32", lanes).build(), inputs)
    for actual, expected in zip(wide[:2], legacy[:2]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected = inputs["streams"][:, 0].float() * (torch.sigmoid(torch.tensor(0.)) + _pre_attrs()["sinkhorn_epsilon"])
    assert not torch.equal(expected, expected.bfloat16().float())
    torch.testing.assert_close(wide[2].reshape_as(expected), expected, atol=0, rtol=0)
    torch.testing.assert_close(wide[2].reshape_as(expected).bfloat16(), legacy[2], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", ["int32", "float64", "float8_e4m3fn", "bad", True, 8])
def test_rejects_unsupported_collapsed_dtypes(dtype):
    with pytest.raises(IRSchemaError, match="collapsed_dtype"):
        MhcPre.normalize_attrs(_pre_attrs(collapsed_dtype=dtype))


@pytest.mark.parametrize("dtype", [None, "bfloat16", "float16", "float32"])
def test_dtype_round_trip_and_default_canonicalization(tmp_path, dtype):
    module = WideCollapse(dtype, (2, 4)).build()
    path = fm.emit_module(module, tmp_path / "pre.py")
    loaded = fm.load_module(path)
    assert loaded.semantic_hash == module.semantic_hash
    if dtype is None:
        assert "collapsed_dtype" not in module.node_map["pre"].attrs
    else:
        assert module.node_map["pre"].attrs["collapsed_dtype"] == dtype


def test_wide_packed_distributed_field_and_cost():
    mesh = fm.Placement((2, 2), "yx", "bb")
    b = fm.SBP.broadcast()
    inputs = tuple(fm.Node(n.id, "builtin.var", (), fm.DistributedType(n.type, (b,) * n.type.rank, mesh),
                           attrs={"name": n.id}) for n in _pre_types())
    attrs = _pre_attrs(collapsed_dtype="float32", output_lanes=(2, 4))
    result = MhcPre.prepare(inputs, attrs).result_type
    expected = MhcPre.prepare(_pre_types(), attrs).result_type
    for field, tensor in zip(result.fields, expected.fields):
        assert field == fm.DistributedType(tensor, (b,) * tensor.rank, mesh)
    narrow = WideCollapse().build().node_map["pre"]
    wide = WideCollapse("float32", (2, 4)).build().node_map["pre"]
    assert MhcPre.cost(wide).bytes_written - MhcPre.cost(narrow).bytes_written == 16 * 2


def test_wide_collapse_python_ir_loads_in_a_fresh_process(tmp_path):
    module = WideCollapse("float32", (2, 4)).build()
    path = fm.emit_module(module, tmp_path / "wide.py")
    code = """from triton.flagmega import ir as fm
import sys
m = fm.load_module(sys.argv[1])
assert m.node_map['pre'].attrs['collapsed_dtype'] == 'float32'
assert m.node_map['pre'].type.fields[2].dtype.elem_type == fm.DType.FLOAT32
print(m.semantic_hash)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == module.semantic_hash
