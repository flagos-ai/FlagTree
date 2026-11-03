"""Packet scheduling policy and LLVM preparation, before native verification.

Wait strategies are alternatives; MMA phase grouping composes with each one.
Preparation is not a correctness certificate: the caller must still verify the
selected native code and recover the original source on preparation/audit failure.
In particular, operand overlap remains a diagnostic with unproved cross-wave WAR.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PacketSchedulePolicy:
    wait: str
    phase: bool
    reuse: str
    cleanup: bool
    coalesce: bool

    @classmethod
    def from_options(cls, options):
        for name in ("mma_phase_schedule", "packet_wait_rotation", "packet_post_mma_wait",
                     "packet_operand_overlap_diagnostic", "packet_redundant_lgkm_cleanup",
                     "coalesce_lgkm_waits"):
            if type(getattr(options, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        strategies = [name for name, enabled in (
            ("rotated", options.packet_wait_rotation),
            ("post_mma", options.packet_post_mma_wait),
            ("overlap_diagnostic", options.packet_operand_overlap_diagnostic)) if enabled]
        if len(strategies) > 1:
            raise ValueError("Choose one packet wait strategy")
        if type(options.packet_reuse_wait) is not str or options.packet_reuse_wait not in ("none", "precopy"):
            raise ValueError("packet_reuse_wait must be none or precopy")
        if options.packet_reuse_wait != "none" and not options.packet_operand_overlap_diagnostic:
            raise ValueError("packet_reuse_wait requires fresh packet_operand_overlap_diagnostic")
        for name in ("packet_operand_overlap_diagnostic", "packet_redundant_lgkm_cleanup",
                     "coalesce_lgkm_waits"):
            if getattr(options, name) and options.arch != "gfx936":
                raise ValueError(f"{name} requires gfx936")
        return cls(strategies[0] if strategies else "full", options.mma_phase_schedule,
                   options.packet_reuse_wait, options.packet_redundant_lgkm_cleanup,
                   options.coalesce_lgkm_waits)


def prepare_packet_schedule(source, metadata, policy, llvm, arch, num_warps):
    """Prepare the chosen wait strategy, then phase grouping, in that order.

    Return whether any preparation changed the source. Metadata and binding
    exceptions are deliberately left to the caller's native verification/fallback.
    A zero-loop result must not replace the source with a returned candidate.
    """
    prepared = False
    if policy.wait != "full":
        prefix, binding_name = {
            "rotated": ("hcu_packet_wait_rotation", "prepare_hcu_packet_wait_rotation"),
            "post_mma": ("hcu_packet_post_mma_wait", "prepare_hcu_packet_post_mma_wait"),
            "overlap_diagnostic": ("hcu_packet_operand_overlap", "prepare_hcu_packet_operand_overlap_diagnostic"),
        }[policy.wait]
        result = getattr(llvm, binding_name)(source, arch, num_warps * 64, int(metadata.get("shared", 0)))
        if policy.wait == "overlap_diagnostic":
            candidate, loops, removed, reason = result
            metadata[prefix + "_removed_waits"] = removed
            metadata[prefix + "_rotation_loops"] = loops
            # The binding proves one removed full wait per fresh group.
            metadata[prefix + "_groups"] = removed
        else:
            candidate, loops, reason = result
        metadata[prefix + "_result"] = reason
        metadata[prefix + "_loops"] = loops
        if loops:
            source, prepared = candidate, True
    if policy.phase:
        candidate, loops = llvm.prepare_hcu_mma_phase_loops(source, arch, num_warps)
        if loops:
            source, prepared = candidate, True
            metadata["hcu_mma_phase_loops"] = loops
    return source, prepared
