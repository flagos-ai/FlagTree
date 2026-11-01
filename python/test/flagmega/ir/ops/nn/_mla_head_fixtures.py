# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from math import prod
from triton.flagmega import ir as fm
from triton.flagmega.ir.distributed_type import scale_split_units


def policies(mesh, heads=8, cyclic=False, staged=False):
    axis = mesh.rank - 1
    if staged:
        head = fm.SBP.split(fm.SplitStage.contiguous((axis,), 4), fm.SplitStage.block_cyclic((0,), 1))
    else:
        head = (
            fm.SBP.split_block_cyclic((axis,), 2)
            if cyclic
            else fm.SBP.split_contiguous((axis,), (heads + mesh.hierarchy[axis] - 1) // mesh.hierarchy[axis])
        )
    return head, scale_split_units(head, 32, 1)


class Projection(fm.Module):
    def __init__(
        self,
        *,
        direction="absorb_query",
        lanes=(2, 4),
        output_lanes=(4,),
        heads=8,
        mesh=None,
        head_policy=None,
        row_policy=None,
        fp8=False,
        scale_sharded=False,
        source_changes=None,
        weight_changes=None,
        scale_changes=None,
        reusable=False,
    ):
        super().__init__(
            dialect="high_level",
            stage="distributed" if mesh else "imported",
            entry="main",
            metadata={"auto_distribution": {"placement": mesh.to_data()}} if mesh else {},
        )
        self.direction, self.lanes, self.output_lanes, self.heads = direction, lanes, output_lanes, heads
        self.mesh, self.head_policy, self.row_policy = mesh, head_policy, row_policy
        self.fp8, self.scale_sharded = fp8, scale_sharded
        self.source_changes, self.weight_changes, self.scale_changes = source_changes, weight_changes, scale_changes
        self.reusable = reusable

    def forward(self):
        width = 16 if self.direction == "absorb_query" else 32
        b = fm.SBP.broadcast()
        kind = fm.vector_type("bfloat16", self.lanes) if self.lanes else "bfloat16"
        x = self.input("x", fm.tensor_type(kind, (1, self.heads, width // prod(self.lanes))), id="x")
        w = self.input("w", fm.tensor_type("float8_e4m3fn" if self.fp8 else "bfloat16", (self.heads * 32, 32)), id="w")
        scale = (
            self.input("scale", fm.tensor_type("float32", (self.heads * 4, 4)), id="scale")
            if self.fp8
            else fm.F.builtin.none()
        )
        parameters = [x, w, *((scale,) if self.fp8 else ())]
        values = [x, w, scale]
        if self.mesh:
            source_policies = [b, self.head_policy or b, b]
            weight_policies = [self.row_policy or b, b]
            scale_policies = [scale_split_units(self.head_policy, 4, 1) if self.scale_sharded else b, b]
            for target, changes in (
                (source_policies, self.source_changes),
                (weight_policies, self.weight_changes),
                (scale_policies, self.scale_changes),
            ):
                if changes:
                    for axis, policy in changes.items():
                        target[axis] = policy
            values = [
                fm.F.distributed.boxing(v, fm.DistributedType(v.type, tuple(p), self.mesh))
                if not isinstance(v.type, fm.NoneType)
                else v
                for v, p in zip(values, (source_policies, weight_policies, scale_policies))
            ]
        result = fm.F.nn.mla_projection(
            *values,
            direction=self.direction,
            qk_head_dim=16,
            value_head_dim=16,
            latent_dim=32,
            weight_block_n=8,
            weight_block_k=8,
            output_lanes=self.output_lanes,
        )
        if self.mesh:
            result = fm.F.distributed.boxing(result, result.type.tensor)
        if self.reusable:
            self.function("project", parameters, (result,), attrs={"reusable": True, "noinline": True})
            outer = [
                self.input("outer_" + name, v.type, id="outer_" + name)
                for name, v in zip(("x", "w", "scale"), parameters)
            ]
            result = fm.F.builtin.call(*outer, callee="project", result_type=result.type)
            parameters = outer
        self.function("main", parameters, (result,))


def graph(**kwargs):
    return Projection(**kwargs).build()
