# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One prepared KDA kernel must honor changing request phases."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

from python.test.flagmega.codegen.triton.kernels.kda_gated_recurrent_core.test_kda_gated_recurrent_runtime import (
    _module,
    _values,
)


def make_runtime(module, path):
    compiled = Compiler().compile(module).module
    assert not any(node.op == "tensors.pack" for node in compiled.nodes)
    artifact = write_artifact(compiled, path, target="nvidia-sm90", emit_executable=True)
    return load(artifact, device="cuda:0")


def bind(runtime, inputs, packed):
    outputs = (torch.empty((1, 2, 16, 8) if packed else (1, 2, 128), dtype=torch.bfloat16, device="cuda"),
               torch.empty((1, 2, 128, 128), dtype=torch.float32, device="cuda"))
    result_names = [argument["value"] for argument in runtime.external_arguments if argument["role"] == "result"]
    values = {**inputs, **dict(zip(result_names, outputs, strict=True))}
    return tuple(values[argument["value"]] for argument in runtime.external_arguments), outputs


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires CUDA")
@pytest.mark.parametrize("packing", ["none", "explicit", "consumer"])
def test_one_prepared_kernel_switches_phase_without_reinterpreting_state(tmp_path, packing):
    packed = packing != "none"
    attrs = {"heads": 2, "dim": 128, "query_scale": 128 ** -.5,
             "output_lanes": (8,) if packing == "explicit" else (),
             "consumer_lanes": (8,) if packing == "consumer" else ()}
    values = _values(torch, device="cuda", heads=2, dim=128)
    phase = True
    runtime = make_runtime(_module(request_phase=True, **attrs), tmp_path / "phase")
    args, actual = bind(runtime, {**values, "prefill": phase}, packed)
    args = list(args)
    phase_index = next(index for index, argument in enumerate(runtime.external_arguments)
                       if argument["value"] == "prefill")
    try:
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        # Keep the same nonzero state and the same prepared input addresses.
        for step, prefill in enumerate((True, False, True)):
            expected_runtime = make_runtime(_module(arithmetic="chunk_bf16" if prefill else "recurrent", **attrs),
                                            tmp_path / f"static-{step}")
            try:
                reference_args, expected = bind(expected_runtime, values, packed)
                GeneratedTirCallGraphModule.prepare(expected_runtime, *reference_args)
                GeneratedTirCallGraphModule.run_into(expected_runtime, *reference_args)
                args[phase_index] = prefill
                GeneratedTirCallGraphModule.run_into(runtime, *args)
                for got, want in zip(actual, expected, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            finally:
                expected_runtime.close()
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


def nested_phase_module():
    placement = fm.Placement((2, 4), "yx", "bb")
    shape = {name: (1, 2, 128) for name in ("q", "k", "v", "gate", "decay_rate", "gate_bias")}
    shape.update(beta=(1, 2), state=(1, 2, 128, 128), prefill=())
    types = {name: fm.tensor_type("bool" if name == "prefill" else
                                  "float32" if name in {"decay_rate", "gate_bias", "state"}
                                  else "bfloat16", extents) for name, extents in shape.items()}
    distributed = {name: fm.DistributedType(kind, (fm.SBP.broadcast(),) * kind.rank, placement)
                   for name, kind in types.items()}

    class Graph(fm.Module):
        def forward(self):
            parameters = tuple(self.input(name, kind, id="step_" + name) for name, kind in distributed.items())
            pair = fm.F.nn.kda_gated_step(*parameters, lower_bound=-5., query_scale=128 ** -.5)
            self.function("step", parameters, fm.F.tensors.get_items(pair, 0, 1),
                          attrs={"reusable": True, "noinline": True})
            values = tuple(self.input(name, kind, id=name) for name, kind in types.items())
            arguments = tuple(fm.F.distributed.force_boxing(value, distributed[name])
                              for name, value in zip(types, values, strict=True))
            result = fm.F.builtin.call(*arguments, callee="step", result_type=pair.type)
            outputs = tuple(fm.F.distributed.force_boxing(value, fm.logical_type(value.type))
                            for value in fm.F.tensors.get_items(result, 0, 1))
            self.function("main", values, outputs)

    return Graph(dialect="distributed", stage="frozen_constants", entry="main",
                 metadata={"auto_distribution": {"placement": placement.to_data()}}).build()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="requires CUDA")
def test_materialized_phase_keeps_scalar_value_across_pooled_function_boundary(tmp_path):
    values = _values(torch, device="cuda", heads=2, dim=128)
    runtime = make_runtime(nested_phase_module(), tmp_path / "nested")
    args, actual = bind(runtime, {**values, "prefill": True}, False)
    args = list(args)
    phase_index = next(index for index, argument in enumerate(runtime.external_arguments)
                       if argument["value"] == "prefill")
    try:
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        for step, prefill in enumerate((True, False, True)):
            reference = make_runtime(_module(heads=2, dim=128, query_scale=128 ** -.5,
                                              arithmetic="chunk_bf16" if prefill else "recurrent"),
                                     tmp_path / f"reference-{step}")
            try:
                reference_args, expected = bind(reference, values, False)
                GeneratedTirCallGraphModule.prepare(reference, *reference_args)
                GeneratedTirCallGraphModule.run_into(reference, *reference_args)
                args[phase_index] = prefill
                GeneratedTirCallGraphModule.run_into(runtime, *args)
                for got, want in zip(actual, expected, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            finally:
                reference.close()
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
