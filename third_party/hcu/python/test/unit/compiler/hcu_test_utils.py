"""Small result adapters for isolated rules of the production LLVM runner."""
from triton._C.libtriton import llvm


def factor_offsets(source, arch):
    output, stats = llvm.optimize_hcu_memory_ir(source, arch, True, False, False)
    return output, stats["factored_stores"]


def pack_rounds(source):
    output, stats = llvm.optimize_hcu_memory_ir(source, "", False, True, False)
    return output, stats["packed_bf16_rounds"]


def combine_shifts(source, arch):
    output, stats = llvm.optimize_hcu_memory_ir(source, arch, False, False, True)
    return output, stats["combined_shifts"], stats["shifted_reason"]


def prepare_schedule(source, arch, threads, shared_bytes):
    output, stats = llvm.optimize_hcu_memory_ir(source, arch, False, False, False, True, threads // 64, shared_bytes)
    return output, stats["post_loops"], stats["post_reason"]
