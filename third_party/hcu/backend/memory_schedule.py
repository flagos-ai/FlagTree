"""One late LLVM transaction; native completion remains mandatory."""


def prepare_memory_schedule(source, metadata, options, llvm, skip_schedule=False):
    schedule = options.memory_mma_schedule and not skip_schedule
    # Legality alone does not prove profitability: store factoring can worsen
    # scheduling, and packed rounds can cross a VGPR occupancy boundary.
    factor = options.buffer_store_factor_offsets
    pack = options.arch == "gfx936"
    combine = options.arch == "gfx936"
    if not (schedule or factor or pack or combine):
        return source, False
    candidate, stats = llvm.optimize_hcu_memory_ir(source, options.arch, factor, pack, combine, schedule,
                                                   options.num_warps, int(metadata.get("shared", 0)))
    if schedule:
        reason = stats["post_reason"]
        metadata.update(hcu_post_mma_wait_preparation_result=reason, hcu_post_mma_wait_result=reason,
                        hcu_post_mma_wait_loops=stats["post_loops"], hcu_mma_phase_loops=stats["phase_loops"])
    if factor:
        metadata["hcu_factored_buffer_store_offsets"] = stats["factored_stores"]
    if pack:
        metadata["hcu_packed_bf16_round_sites"] = stats["packed_bf16_rounds"]
    if combine:
        metadata["hcu_shifted_disjoint_or_sites"] = stats["combined_shifts"]
        metadata["hcu_shifted_disjoint_or_result"] = stats["shifted_reason"]
    prepared = bool(stats["post_loops"] or stats["phase_loops"])
    changed = prepared or any(stats[k] for k in ("factored_stores", "packed_bf16_rounds", "combined_shifts"))
    return candidate if changed else source, prepared
