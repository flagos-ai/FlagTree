# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
from dataclasses import replace
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia.mhc import (
    mhc_cooperative_stream_fma_implementations,
    mhc_cooperative_contiguous_tf32_implementations,
)


def compile_graph(graph, independent=None, *, declared=None):
    mesh = graph.metadata["auto_distribution"]["placement"]
    from triton.flagmega import ir as fm

    placement = fm.Placement(tuple(mesh["hierarchy"]), mesh["name"], mesh["hierarchy_levels"])
    target = NvidiaSm90Target()
    base = target.triton_implementation_model
    extra = (*mhc_cooperative_stream_fma_implementations(), *mhc_cooperative_contiguous_tf32_implementations())
    extra = tuple(
        replace(
            i,
            parameters={
                **i.parameters,
                **({"independent_collapse_pre": independent} if independent is not None else {}),
            },
            facts={
                **i.facts,
                "internal_grid_barriers": declared if declared is not None else 1 if independent is True else 2,
            },
        )
        for i in extra
    )
    target.triton_implementation_model = replace(
        base,
        implementations=(*base.implementations, *extra),
        preferences={**base.preferences, "mhc_pre": tuple(i.id for i in extra)},
    )
    c = Compiler(CompileOptions(placement=placement))
    c.target = target
    return c.compile(graph).module
