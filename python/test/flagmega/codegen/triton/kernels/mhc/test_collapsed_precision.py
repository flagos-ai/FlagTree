# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Execute BF16 streams -> FP32 collapse -> FP32 stats / BF16 numerator."""

from math import prod

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load

from python.test.flagmega.codegen.triton.kernels.mhc.test_mhc_runtime import _artifact, _run, _skip_without_sm90
from python.test.flagmega.ir.ops.nn.test_mhc import _pre_attrs
from python.test.flagmega.ir.ops.nn.test_mhc_collapsed_dtype import WideCollapse, rounding_inputs


@pytest.mark.parametrize("lanes", [(), (8,), (2, 4)])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_bf16_streams_keep_the_declared_collapsed_precision_on_device(tmp_path, lanes, dtype):
    _skip_without_sm90(torch)
    host = rounding_inputs()
    module = WideCollapse(dtype, lanes).build()
    artifact = _artifact(tmp_path, "wide-collapse", module)
    source = (artifact / "generated_kernels.py").read_text()
    assert source.count("# flagmega-kernel: mhc_pre/local") == 1
    assert "# flagmega-kernel: pack/" not in source
    assert "# flagmega-kernel: tensor_transform/" not in source
    runtime = load(artifact, device="cuda:0")
    try:
        values = {name: value.cuda() for name, value in host.items()}
        outputs = {"field_0": torch.empty((1, 4), dtype=torch.float32, device="cuda"),
                   "field_1": torch.empty((1, 4, 4), dtype=torch.float32, device="cuda"),
                   "field_2": torch.empty((1, 16 // prod(lanes), *lanes), dtype=getattr(torch, dtype), device="cuda")}
        _run(runtime, values, outputs, torch)
        expected = host["streams"][:, 0].float() * (torch.sigmoid(torch.tensor(0.)) + _pre_attrs()["sinkhorn_epsilon"])
        assert not torch.equal(expected, expected.bfloat16().float())
        torch.testing.assert_close(outputs["field_2"].cpu().reshape_as(expected), expected.to(getattr(torch, dtype)),
                                   atol=0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


class SplitNormPrecision(WideCollapse):
    def forward(self):
        types = {"streams": fm.tensor_type("bfloat16", (1, 4, 16)),
                 "projection": fm.tensor_type("float32", (24, 64)),
                 "base": fm.tensor_type("float32", (24,)), "scale": fm.tensor_type("float32", (3,)),
                 "norm_weight": fm.tensor_type("float32", (16,)), "bias": fm.tensor_type("float32", (16,))}
        values = {name: self.input(name, value, id=name) for name, value in types.items()}
        pre = fm.F.nn.mhc_pre(*(values[k] for k in ("streams", "projection", "base", "scale")),
                              **_pre_attrs(), collapsed_dtype="float32", name="pre")
        collapsed = fm.F.tensors.get_item(pre, 2, name="collapsed")
        stats = fm.F.nn.norm_stats(collapsed, axis=-1, use_mean=False, name="stats")
        numerator = fm.F.tensors.cast(collapsed, "bfloat16", name="numerator")
        normalized = fm.F.nn.norm_apply(numerator, stats, values["norm_weight"], values["bias"],
                                       axis=-1, epsilon=1e-5, use_mean=False, output_dtype="bfloat16", name="normalized")
        self.function("main", tuple(values.values()), (collapsed, stats, numerator, normalized))


def test_norm_uses_unrounded_statistics_and_a_rounded_numerator(tmp_path):
    _skip_without_sm90(torch)
    module = SplitNormPrecision().build()
    host = {**rounding_inputs(), "norm_weight": torch.ones(16), "bias": torch.zeros(16)}
    expected = TorchEvaluator(DictWeightResolver({})).run(module, host)
    raw = host["streams"][:, 0].float() * (torch.sigmoid(torch.tensor(0.)) + _pre_attrs()["sinkhorn_epsilon"])
    rounded_stats = raw.bfloat16().float().square().sum().reshape(1, 1, 1)
    assert float((expected[1] - rounded_stats).abs().max()) > 1e-5
    artifact = _artifact(tmp_path, "split-norm", module)
    runtime = load(artifact, device="cuda:0")
    try:
        values = {name: value.cuda() for name, value in host.items()}
        names = runtime.ir_module.function_map[runtime.ir_module.entry].outputs
        outputs = {name: torch.empty_like(value, device="cuda") for name, value in zip(names, expected, strict=True)}
        _run(runtime, values, outputs, torch)
        for index, (name, value) in enumerate(zip(outputs, expected, strict=True)):
            torch.testing.assert_close(outputs[name].cpu(), value, atol=1e-5 if index == 1 else 0, rtol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
