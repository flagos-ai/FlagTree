# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Exclusive data ownership must not remove grid-barrier participants."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.kernel_call_renderers import _exclusive_participation_active


@pytest.mark.parametrize("barriers", [0, 1, 2])
@pytest.mark.parametrize(
    "levels,axes,expected",
    [
        ("bb", (0, 1), "True"),
        ("cbb", (1, 2), "True"),
        ("cbb", (0, 1, 2), "(shard_coord0 == 1)"),
        ("bdb", (0, 1, 2), "(shard_coord1 == 1)"),
    ],
)
def test_internal_grid_rendezvous_retains_every_block_in_selected_chips(barriers, levels, axes, expected):
    rank = len(levels)
    raw = {
        "facts": {"internal_grid_barriers": barriers},
        "inputs": [
            {
                "buffers": [
                    {
                        "abi": {
                            "distributed_type": {
                                "placement": {"hierarchy": (4,) * rank, "hierarchy_levels": levels},
                                "exclusive": {"axes": axes, "owner_coordinates": (1,) * len(axes)},
                            }
                        }
                    }
                ]
            }
        ],
    }
    actual = _exclusive_participation_active(raw)
    if barriers:
        assert actual == expected
    else:
        assert actual.count("== 1") == len(axes)


def _run_cache_pipeline(directory, schedule):
    from dataclasses import replace

    import torch
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    torch.set_num_threads(4)
    mesh = fm.Placement((8, 16), "yx", "bb")
    config = PagedAttentionStateConfig(1, 1, 32)

    class Graph(fm.Module):
        def forward(self):
            kinds = {
                "q": fm.tensor_type("bfloat16", (1, 2, 16)),
                "c": fm.tensor_type("bfloat16", (1, 32)),
                "w": fm.tensor_type("bfloat16", (64, 32)),
                "topk": fm.tensor_type("int32", (1, 2)),
                "state": config.ref_type,
                "sentinel": fm.tensor_type("float32", (128,)),
            }
            raw = {name: self.input(name, kind, id=name) for name, kind in kinds.items()}
            values = {
                name: fm.F.distributed.force_boxing(
                    raw[name],
                    fm.DistributedType(
                        kinds[name],
                        (fm.SBP.broadcast(),) * kinds[name].rank,
                        mesh,
                        exclusive=fm.SBP.exclusive((0, 1)) if name == "c" else None,
                    ),
                )
                for name in ("q", "c", "w", "topk")
            }
            pair = fm.F.tensors.reshape(fm.F.tensors.concat(values["c"], values["c"], axis=1), (1, 1, 64))
            layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
            advance = fm.F.builtin.scalar_const(fm.tensor_type("bool", ()), True)
            state = fm.F.nn.update_paged_attention_kv_cache_pair(
                pair, raw["state"], layer, advance, layout=("seq", "head", "dim")
            )
            attrs = {"qk_head_dim": 16, "value_head_dim": 16, "latent_dim": 32}
            ql = fm.F.nn.mla_projection(
                values["q"], values["w"], fm.F.builtin.none(), direction="absorb_query", **attrs
            )
            latent = fm.F.nn.dsa_paged_attention(
                ql,
                values["topk"],
                state,
                layer,
                scale=0.25,
                layout=("seq", "head", "dim"),
                hidden_size=64,
                probability_rounding="online_bf16",
            )
            context = fm.F.nn.mla_projection(
                latent, values["w"], fm.F.builtin.none(), direction="expand_value", **attrs
            )
            context = fm.F.distributed.force_boxing(context, fm.logical_type(context.type))
            split = fm.DistributedType(kinds["sentinel"], (fm.SBP.split_contiguous((0, 1), 1),), mesh)
            local = fm.F.distributed.force_boxing(raw["sentinel"], split)
            aux = fm.F.distributed.force_boxing(fm.F.math.add(local, local), kinds["sentinel"])
            self.function("main", tuple(raw.values()), (context, aux))

    source = Graph(
        dialect="distributed",
        stage="frozen_constants",
        entry="main",
        metadata={"auto_distribution": {"placement": mesh.to_data()}},
    ).build()
    compiler = Compiler()
    if schedule == "bf16_mma":
        from triton.flagmega.targets import NvidiaSm90Target

        base = compiler.target.triton_implementation_model
        catalog = replace(
            base,
            implementations=tuple(
                replace(value, parameters={**value.parameters, "decode_schedule": schedule, "compute_num_warps": 8})
                if value.id == "tir.dsa_paged_attention.decode"
                else value
                for value in base.implementations
            ),
        )
        compiler.target = NvidiaSm90Target(triton_implementation_model=catalog)
    compiled = compiler.compile(source).module
    artifact = write_artifact(compiled, directory / "artifact", target="nvidia-sm90", emit_executable=True)
    runtime = load(artifact, device="cuda:0")
    state = create_paged_attention_state(config, device="cuda")
    state.seq_lens.fill_(1)
    state.slot_mapping.fill_(1)
    for kind in (0, 1):
        state.kv_caches[0, 0, kind, 0, 0].fill_(1)
    values = {
        "q": torch.zeros((1, 2, 16), dtype=torch.bfloat16, device="cuda"),
        "c": torch.full((1, 32), 2, dtype=torch.bfloat16, device="cuda"),
        "w": torch.ones((64, 32), dtype=torch.bfloat16, device="cuda"),
        "topk": torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        "sentinel": torch.arange(128, dtype=torch.float32, device="cuda"),
    }
    buffers = {}
    for value, ids in runtime.buffer_plan.function_map[compiled.entry].parameters:
        if isinstance(compiled.node_map[value].type, fm.RefType):
            buffers.update(
                zip(
                    ids,
                    (state.kv_caches, state.query_start_loc, state.seq_lens, state.slot_mapping, state.block_table),
                    strict=True,
                )
            )
        else:
            buffers[ids[0]] = values[compiled.node_map[value].attrs["name"]]
    context = torch.empty((1, 2, 16), dtype=torch.bfloat16, device="cuda")
    aux = torch.empty(128, dtype=torch.float32, device="cuda")
    for (_, ids), output in zip(runtime.buffer_plan.function_map[compiled.entry].outputs, (context, aux), strict=True):
        buffers[ids[0]] = output
    args = tuple(buffers[spec["buffer"]] for spec in runtime.external_arguments)
    try:
        GeneratedTirCallGraphModule.prepare(runtime, *args)
        print(runtime.resource_report, flush=True)
        GeneratedTirCallGraphModule.run_into(runtime, *args)
        torch.cuda.synchronize()
        torch.testing.assert_close(context, torch.full_like(context, 48), rtol=0, atol=0)
        torch.testing.assert_close(aux, values["sentinel"] * 2, rtol=0, atol=0)
        assert state.seq_lens.item() == 2
        assert runtime.resource_report["grid"] == [128, 1, 1]
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("schedule", ["simt", "bf16_mma"])
def test_exclusive_cache_update_rendezvous_then_latent_attention(tmp_path, schedule):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires CUDA")
    # Bound the child lifetime: the regression was a device barrier deadlock.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), schedule],
        env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _run_cache_pipeline(Path(sys.argv[1]), sys.argv[2])
