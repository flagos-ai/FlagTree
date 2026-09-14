"""H800 single-token expert/GDN tile experiments with the normal target catalog."""

from dataclasses import replace

from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


def create_target(*, gate_n=8, gate_k=128, down_n=8, down_k=128, gdn_value_tile=None, gdn_projection_tile=None,
                  bufferize_opt_level="optimized"):
    tiles = {"sparse_experts_gate_up": (gate_n, gate_k), "sparse_experts_down": (down_n, down_k)}
    checked_tiles = (*[v for values in tiles.values() for v in values],
                     *(v for v in (gdn_value_tile, gdn_projection_tile) if v is not None))
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v & (v - 1)
           for v in checked_tiles):
        raise ValueError("Tiles must be positive integer powers of two")
    model = sm90_triton_implementation_model()

    def configure(implementation):
        if implementation.family == "gdn_recurrent" and implementation.variant == "persistent":
            parameters = dict(implementation.parameters)
            if gdn_value_tile is not None:
                parameters["tile_state"] = (parameters["tile_state"][0], gdn_value_tile)
            if gdn_projection_tile is not None:
                parameters["projection_tile"] = gdn_projection_tile
            return replace(implementation, parameters=parameters)
        if implementation.family not in tiles or implementation.variant != "simt":
            return implementation
        block_n, block_k = tiles[implementation.family]
        return replace(implementation, parameters={**implementation.parameters, "block_n": block_n, "block_k": block_k})

    target = NvidiaSm90Target(triton_implementation_model=replace(
        model, implementations=tuple(configure(value) for value in model.implementations)))
    return target.with_bufferize_opt_level(bufferize_opt_level)
