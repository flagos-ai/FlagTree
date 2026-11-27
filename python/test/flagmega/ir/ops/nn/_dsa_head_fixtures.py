# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace
from math import prod
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig


CONFIG = PagedAttentionStateConfig(1, 1, 32, num_blocks=8)


class Attention(fm.Module):
    def __init__(
        self,
        *,
        mesh,
        head_policy=None,
        heads=8,
        lanes=(2, 4),
        traversal="compact",
        query_changes=None,
        topk_policy=None,
        config=CONFIG,
    ):
        super().__init__(
            dialect="high_level",
            stage="distributed",
            entry="main",
            metadata={"auto_distribution": {"placement": mesh.to_data()}},
        )
        self.mesh, self.hp, self.heads, self.lanes, self.traversal = mesh, head_policy, heads, lanes, traversal
        self.query_changes, self.topk_policy = query_changes, topk_policy
        self.config = config

    def forward(self):
        b = fm.SBP.broadcast()
        dtype = fm.vector_type("bfloat16", self.lanes) if self.lanes else "bfloat16"
        query = self.input("query", fm.tensor_type(dtype, (1, self.heads, 32 // prod(self.lanes))), id="query")
        topk = self.input("topk", fm.tensor_type("int32", (1, 130)), id="topk")
        state = self.input("state", self.config.ref_type, id="state")
        layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 0)
        qp = [b, self.hp or b, b]
        for axis, policy in (self.query_changes or {}).items():
            qp[axis] = policy
        q = fm.F.distributed.boxing(query, fm.DistributedType(query.type, tuple(qp), self.mesh))
        k = fm.F.distributed.boxing(topk, fm.DistributedType(topk.type, (b, self.topk_policy or b), self.mesh))
        result = fm.F.nn.dsa_paged_attention(
            q,
            k,
            state,
            layer,
            scale=0.25,
            layout=("seq", "head", "dim"),
            hidden_size=self.heads * 32,
            probability_rounding="online_bf16",
            index_traversal=self.traversal,
        )
        result = fm.F.distributed.boxing(result, result.type.tensor)
        self.function("main", (query, topk, state), (result,))


def graph(**kwargs):
    return Attention(**kwargs).build()


def compile_attention(path, *, schedule="bf16_mma", **kwargs):
    c = Compiler(CompileOptions(placement=kwargs["mesh"]))
    base = c.target.triton_implementation_model
    c.target = NvidiaSm90Target(
        triton_implementation_model=replace(
            base,
            implementations=tuple(
                replace(i, parameters={**i.parameters, "decode_schedule": schedule, "compute_num_warps": 8})
                if i.id == "tir.dsa_paged_attention.decode"
                else i
                for i in base.implementations
            ),
        )
    )
    m = c.compile(graph(**kwargs)).module
    r = fm.load_module(fm.emit_module(m, path / "compiled.py"))
    assert r.semantic_hash == m.semantic_hash
    return r, write_artifact(r, path / "artifact", target="nvidia-sm90", emit_executable=True)
