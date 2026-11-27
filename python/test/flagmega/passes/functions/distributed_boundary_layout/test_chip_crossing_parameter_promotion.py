# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Do not promote a parameter whose in-callee restore would read peer chips.

Promoting a reusable callee's parameter to a chip-split ABI moves the reshard
work to the caller, which is normally the cheaper plan.  But a raw use inside
the callee then needs the logical value back, and rebuilding it from a
chip-split parameter is an all-gather: every chip reads the other chips' copies
of that parameter.

A parameter is an opaque pointer the caller supplies, so its peer-chip copies
have no address the callee can form; only pool storage is registered as the
FlagCX symmetric window.  The planner has to see that before it commits to the
promotion, otherwise the module type-checks and only fails in codegen when
``emit_remote_storage_pointer`` looks for the peer base.
"""

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.passes.functions import (
    propagate_post_auto_distributed_function_boundary_layouts,
)

# ``cbb``: axis 0 spans chips, axes 1 and 2 are block axes inside one chip.
_PLACEMENT = fm.Placement((4, 2, 4), "cyx", "cbb")
_TENSOR = fm.tensor_type("float32", (1, 64))
_BROADCAST = fm.DistributedType(
    _TENSOR, (fm.SBP.broadcast(), fm.SBP.broadcast()), _PLACEMENT)
_CHIP_SPLIT = fm.DistributedType(
    _TENSOR, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 16)), _PLACEMENT)
_BLOCK_SPLIT = fm.DistributedType(
    _TENSOR, (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 32)), _PLACEMENT)


def _module(demanded):
    """A reusable callee with two adapters demanding ``demanded``, plus a raw use.

    Two adapters outvote the single raw use, so the planner's internal variant
    promotes the parameter to ``demanded`` and restores the logical value for
    the raw use.
    """

    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="distributed", entry="main")

        def forward(self):
            hidden = self.input("hidden", _BROADCAST, id="hidden")
            first = fm.F.distributed.boxing(hidden, demanded, name="first")
            second = fm.F.distributed.boxing(hidden, demanded, name="second")
            combined = fm.F.math.add(first, second, name="combined")
            raw = fm.F.math.silu(hidden, name="raw")
            output = fm.F.math.add(raw, fm.F.distributed.boxing(
                combined, _BROADCAST, name="restore"), name="output")
            value = self.input("value", _BROADCAST, id="value")
            call = fm.F.builtin.call(
                value, result_type=_BROADCAST, callee="layer", name="call")
            self.function("main", (value,), (call,))
            self.function(
                "layer", (hidden,), (output,),
                attrs={"calling_convention": "device", "noinline": True,
                       "reusable": True},
            )

    return Graph().build()


def _restores(module):
    return tuple(
        node for node in module.nodes
        if node.metadata.get("boundary_layout") == "raw_parameter_restore"
    )


def test_a_chip_crossing_restore_blocks_the_promotion():
    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        _module(_CHIP_SPLIT), PyNttDistributedReshardRealizationPolicy())
    parameter = rewritten.node_map[rewritten.function_map["layer"].parameters[0]]
    assert parameter.type == _BROADCAST
    # No restore exists, because nothing was promoted away from the logical type.
    assert _restores(rewritten) == ()


def test_a_block_only_restore_still_promotes():
    """The guard is about peer chips, not about promotion in general.

    The same graph with a block-split demand keeps every owner inside one chip
    instance, so the restore is chip-local and the promotion is still taken.
    """

    rewritten = propagate_post_auto_distributed_function_boundary_layouts(
        _module(_BLOCK_SPLIT), PyNttDistributedReshardRealizationPolicy())
    parameter = rewritten.node_map[rewritten.function_map["layer"].parameters[0]]
    assert parameter.type == _BLOCK_SPLIT
    restore, = _restores(rewritten)
    assert restore.inputs == (parameter.id,)
    assert restore.type == _BROADCAST
    assert fm.boxing_crossing_chip_axes(parameter.type, restore.type) == ()


def test_no_surviving_restore_reads_a_peer_chip_instance():
    """The property codegen depends on, stated directly.

    ``emit_remote_storage_pointer`` needs pool storage for a peer read, and a
    parameter never has it.  Assert the invariant over both graphs rather than
    only over the promotion decision, so a future planner change that
    reintroduces such a restore fails here instead of in codegen.
    """

    for demanded in (_CHIP_SPLIT, _BLOCK_SPLIT):
        rewritten = propagate_post_auto_distributed_function_boundary_layouts(
            _module(demanded), PyNttDistributedReshardRealizationPolicy())
        for restore in _restores(rewritten):
            if restore.op != "distributed.boxing":
                continue
            source = rewritten.node_map[restore.inputs[0]]
            assert fm.boxing_crossing_chip_axes(source.type, restore.type) == ()
