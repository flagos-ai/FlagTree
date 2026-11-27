# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace
from pathlib import Path
from math import prod
import pytest, torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.runtime import load
from triton.flagmega.ir.ops.nn.mhc_pre import MhcPre
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Run
from python.test.flagmega.codegen.triton.kernels.mhc.test_cooperative_stream_fma import distributed_module
from python.test.flagmega.ir.ops.nn.test_mhc_ordered_projection import module
from python.test.flagmega.codegen.triton.kernels.mhc._independent_pre_fixture import compile_graph

root = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "ctas,tokens,hc,hidden,splits,reduction,collapse",
    [
        (128, 1, 4, 4096, 8, "stream_fma_128", "sequential_fma"),
        (128, 3, 3, 512, 2, "stream_fma_128", "sum"),
        (1, 3, 3, 512, 2, "stream_fma_128", "sequential_fma"),
        (1, 2, 1, 128, 1, "stream_fma_128", "sum"),
        (128, 1, 4, 4096, 64, "contiguous_tf32", "sequential_fma"),
        (1, 1, 4, 4096, 64, "contiguous_tf32", "sum"),
    ],
)
def test_independent_pre_preserves_complete_outputs(tmp_path, ctas, tokens, hc, hidden, splits, reduction, collapse):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90")
    mesh = fm.Placement((8, 16) if ctas == 128 else (1, 1), "yx", "bb")
    builder = distributed_module if ctas == 128 else module
    g = builder(
        tokens=tokens,
        hc=hc,
        hidden=hidden,
        splits=8 if reduction == "contiguous_tf32" else splits,
        output_lanes=(4,),
        collapsed_dtype="float32",
        collapse_input_dtype="bfloat16",
        collapse_reduction=collapse,
        **({"placement": mesh} if ctas == 128 else {}),
    )
    g = replace(
        g,
        dialect="distributed",
        stage="frozen_constants",
        metadata={**g.metadata, "auto_distribution": {"placement": mesh.to_data()}},
        nodes=tuple(
            replace(
                n,
                attrs=MhcPre.ir_attrs(
                    MhcPre.normalize_attrs(
                        {
                            **n.attrs,
                            "sinkhorn_iterations": 20,
                            "projection_reduction": reduction,
                            "projection_splits": splits,
                        }
                    )
                ),
            )
            if n.op == "nn.mhc_pre"
            else n
            for n in g.nodes
        ),
    )
    fm.verify_module(g)
    runtimes = {}
    outputs = {}
    shared = {}
    args = {}
    guards = []

    def allocate(kind):
        t = fm.logical_type(kind)
        shape = tuple(d.fixed_value for d in t.shape) + tuple(getattr(t.dtype, "lanes", ()))
        dtype = getattr(torch, getattr(t.dtype, "elem_type", t.dtype).value)
        storage = torch.full((prod(shape) + 32,), 123, dtype=dtype, device="cuda")
        guards.append(storage)
        return storage[16:-16].reshape(shape)

    try:
        for label, mode in [("control", None), ("independent", True)]:
            m = compile_graph(g, mode)
            r = load(write_artifact(m, tmp_path / label, target="nvidia-sm90", emit_executable=True), device="cuda:0")
            runtimes[label] = r
            values = {}
            outputs[label] = {}
            for spec in r.external_arguments:
                name = spec["value"]
                kind = m.node_map[name].type
                if spec["role"] == "result":
                    value = allocate(kind)
                    outputs[label][name] = value
                else:
                    if name not in shared:
                        shared[name] = allocate(kind)
                    value = shared[name]
                values[name] = value
            args[label] = tuple(values[s["value"]] for s in r.external_arguments)
            Run.prepare(r, *args[label])
            assert r.resource_report["spill_bytes"] == 0
            assert r.resource_report["grid"] == [ctas, 1, 1]
        for mutation in range(6):
            for name, value in shared.items():
                if name == "sentinel":
                    value.copy_(torch.arange(128, device="cuda"))
                elif name == "scale":
                    value.copy_(torch.tensor([0.7, 1.1, 0.9], device="cuda"))
                elif name == "projection":
                    value.copy_(torch.randn_like(value) * 0.02)
                else:
                    value.copy_(torch.randn_like(value))
            if mutation == 5:
                shared["streams"].zero_()
                shared["projection"].zero_()
                shared["base"].zero_()
            for label, r in runtimes.items():
                Run.run_into(r, *args[label])
            torch.cuda.synchronize()
            for name, old in outputs["control"].items():
                assert torch.equal(old.view(torch.uint8), outputs["independent"][name].view(torch.uint8)), (
                    mutation,
                    name,
                )
            assert all(bool((g[:16] == 123).all()) and bool((g[-16:] == 123).all()) for g in guards)
    finally:
        for r in runtimes.values():
            r.close()
