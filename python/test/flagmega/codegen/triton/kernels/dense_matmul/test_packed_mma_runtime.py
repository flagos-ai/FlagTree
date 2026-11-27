# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Packed BF16 operands retain Tensor Core accumulation through codegen."""

import ast
from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import DirectoryCheckpoint
from triton.flagmega.passes.auto_distributed import AutoDistributedPass, solve_search_graph
from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
from triton.flagmega.passes.auto_distributed.proposal_analysis import remember_proposal
from triton.flagmega.runtime import load
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
from triton.flagmega.targets import NvidiaSm90Target


class CanonicalTarget(NvidiaSm90Target):
    """Exercise the canonical kernel with an explicit legal distribution."""

    def propose_distribution(self, module):
        graph = AutoDistributedPass._build_graph(module, self)

        def broadcast(value):
            return (not isinstance(value, fm.DistributedType)
                    or (value.partial is None and value.exclusive is None
                        and all(p == fm.SBP.broadcast() for p in value.axis_policies)))

        fixed = {}
        for name, bucket in graph.bucket_map.items():
            if module.node_map[name].op != "ntt.packed_matmul":
                continue
            matches = [c for c in bucket.candidates
                       if broadcast(c.return_type) and all(broadcast(t) for t in c.input_types)]
            assert len(matches) == 1
            fixed[name] = matches[0].id
        solved = solve_search_graph(graph, fixed_selections=fixed)
        points, records = distribution_selection_state(solved, policy=self.distribution_policy.identity)
        old = {p.id for p in module.selection_points if p.kind == "distribution"}
        result = fm.verify_module(replace(module,
            selection_points=tuple(p for p in module.selection_points if p.id not in old) + points,
            selections=tuple(r for r in module.selections if r.point_id not in old) + records))
        remember_proposal(result, graph, self)
        return result


@pytest.mark.parametrize("m,k,n,large", [(1, 48, 24, False), (2, 256, 72, False),
                                      (1, 4096, 1024, False), (1, 4096, 16, True)])
def test_packed_bf16_mma_matches_cuda_projection(tmp_path, m, k, n, large):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA is required")
    from safetensors.torch import save_file

    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (m, k)), id="x")
            w = self.weight("weight", fm.tensor_type("bfloat16", (n, k)),
                            source="model.safetensors", key="weight", id="weight")
            y = fm.F.math.matmul(x, w, transpose_b=True)
            self.function("main", (x,), (y,))

    torch.manual_seed(129)
    weight = torch.full((n, k), 1e17).bfloat16() if large else torch.randn(n, k).bfloat16()
    value = (torch.full((m, k), 1e17).bfloat16() if large else torch.randn(m, k).bfloat16()).cuda()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    save_file({"weight": weight}, checkpoint / "model.safetensors")
    base = NvidiaSm90Target().triton_implementation_model
    identity = "tir.dense_matmul.packed_k_major_mma"
    assert identity in {i.id for i in base.implementations}
    model = replace(base, preferences={**base.preferences, "dense_matmul":
                    (identity, *base.preferences["dense_matmul"])})
    compiler = Compiler()
    compiler.target = CanonicalTarget(triton_implementation_model=model)
    result = compiler.compile(Graph(dialect="high_level", stage="imported", entry="main").build()).module
    artifact = write_artifact(result, tmp_path / "artifact", target="nvidia-sm90",
                              checkpoint=DirectoryCheckpoint(checkpoint), emit_executable=True)
    source = (artifact / "generated_kernels.py").read_text()
    assert "# flagmega-kernel: dense_matmul/packed_k_major_mma" in source
    assert "# flagmega-kernel: pack/" not in source
    projection, = [function for function in ast.parse(source).body
                   if isinstance(function, ast.FunctionDef) and any(
                       isinstance(node, ast.Name) and node.id == "dense_activation_matrix"
                       for node in ast.walk(function))]
    # Entry scheduling values must not span an inlined projection body.
    assert any(isinstance(decorator, ast.Call) and any(
        keyword.arg == "noinline" and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True for keyword in decorator.keywords
    ) for decorator in projection.decorator_list)
    runtime = load(artifact, device="cuda:0")
    try:
        output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        args = tuple(output if s["role"] == "result" else value for s in runtime.external_arguments)
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        GeneratedTirCallGraphModule.run_into(runtime, *args)
        torch.testing.assert_close(output, value @ weight.cuda().T, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
