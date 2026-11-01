# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from math import prod
from dataclasses import replace
import torch
from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.tensors.pack import pack_physical


class SharedModule(fm.Module):
    def __init__(
        self,
        config,
        *,
        shared_kv=True,
        lanes=(),
        tokens=3,
        layout=("seq", "head", "dim"),
        advance=True,
        placement=None,
        split_axis=None,
        cyclic=False,
        reusable=False,
    ):
        super().__init__(
            dialect="high_level",
            stage="distributed" if placement else "imported",
            entry="main",
            metadata={"auto_distribution": {"placement": placement.to_data()}} if placement else {},
        )
        self.config, self.shared_kv, self.lanes, self.tokens, self.layout = config, shared_kv, lanes, tokens, layout
        self.advance, self.placement, self.split_axis, self.cyclic, self.reusable = (
            advance,
            placement,
            split_axis,
            cyclic,
            reusable,
        )

    def forward(self):
        c = self.config
        layout = self.layout
        extents = {"seq": self.tokens, "head": c.num_kv_heads, "dim": c.head_dim // prod(self.lanes)}
        dtype = fm.vector_type("bfloat16", self.lanes) if self.lanes else "bfloat16"
        slots = self.input("slots", fm.tensor_type(dtype, tuple(extents[a] for a in layout)), id="slots")
        source = slots
        # The legacy assembly requires a broadcast concatenation axis.
        # Assemble before the explicit owner assignment, preserving the same
        # final dim shard that the shared-input consumer accepts directly.
        if self.shared_kv is None or self.shared_kv is False:
            slots = fm.F.tensors.concat(slots, slots, axis=layout.index("dim"), name="pair")
        if self.placement:
            b = fm.SBP.broadcast()
            policies = [b] * 3
            if self.split_axis:
                axis = self.placement.hierarchy_levels.index("b")
                policies[layout.index(self.split_axis)] = (
                    fm.SBP.split_block_cyclic((axis,), 2) if self.cyclic else fm.SBP.split_contiguous((axis,))
                )
            slots = fm.F.distributed.boxing(slots, fm.DistributedType(slots.type, tuple(policies), self.placement))
        state = self.input("state", c.ref_type, id="state")
        layer = fm.F.builtin.scalar_const(fm.tensor_type("int32", ()), 1)
        advance = fm.F.builtin.scalar_const(fm.tensor_type("bool", ()), self.advance)
        attrs = {} if self.shared_kv is None else {"shared_kv": self.shared_kv}
        updated = fm.F.nn.update_paged_attention_kv_cache_pair(
            slots, state, layer, advance, layout=layout, **attrs, name="update"
        )
        parameters = (source, state)
        if self.reusable:
            self.function("publish", parameters, (updated,), attrs={"reusable": True, "noinline": True})
            outer_slots = self.input("outer_slots", source.type, id="outer_slots")
            outer_state = self.input("outer_state", state.type, id="outer_state")
            updated = fm.F.builtin.call(
                outer_slots,
                outer_state,
                callee="publish",
                result_type=c.ref_type,
                effect=fm.effect("read_write", c.ref_type.name),
            )
            parameters = (outer_slots, outer_state)
        self.function("main", parameters, (fm.F.builtin.tuple(updated),))


def physical(config, tokens, lanes, layout, device="cpu"):
    semantic = (
        torch.arange(tokens * config.num_kv_heads * config.head_dim, device=device)
        .reshape(tokens, config.num_kv_heads, config.head_dim)
        .bfloat16()
    )
    result = semantic.permute(tuple(("seq", "head", "dim").index(a) for a in layout)).contiguous()
    if lanes:
        result = pack_physical(result, 3, lanes, (layout.index("dim"),) * len(lanes))
    return semantic, result
