"""Late memory/MMA scheduling preparation; native completion stays mandatory."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySchedulePolicy:
    post_mma: bool
    phase: bool

    @classmethod
    def from_options(cls, options):
        for name in ("mma_phase_schedule", "post_mma_wait", "coalesce_lgkm_waits"):
            if type(getattr(options, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if options.coalesce_lgkm_waits and options.arch != "gfx936":
            raise ValueError("coalesce_lgkm_waits requires gfx936")
        return cls(options.post_mma_wait, options.mma_phase_schedule)


def prepare_memory_schedule(source, metadata, policy, llvm, arch, num_warps):
    """Prepare full LDS completion after MMA, then independent MMA grouping.

    Preparation is not a native correctness certificate. The caller verifies
    final code and restores its unscheduled source on preparation/audit failure.
    A zero-loop result must not replace source with a returned candidate.
    """
    prepared = False
    if policy.post_mma:
        candidate, loops, reason = llvm.prepare_hcu_post_mma_wait(
            source, arch, num_warps * 64, int(metadata.get("shared", 0)))
        metadata["hcu_post_mma_wait_preparation_result"] = reason
        metadata["hcu_post_mma_wait_result"] = reason
        metadata["hcu_post_mma_wait_loops"] = loops
        if loops:
            source, prepared = candidate, True
    # A requested pair must not degrade to phase-only motion when the
    # prerequisite wait proof declines the loop. Phase-only remains available
    # explicitly with post_mma=False.
    if policy.phase and (not policy.post_mma or prepared):
        candidate, loops = llvm.prepare_hcu_mma_phase_loops(source, arch, num_warps)
        if loops:
            source, prepared = candidate, True
            metadata["hcu_mma_phase_loops"] = loops
    return source, prepared
