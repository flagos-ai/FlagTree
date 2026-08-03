#ifndef MK_PIPELINE_PASSES
#define MK_PIPELINE_PASSES

include "mlir/Pass/PassBase.td"

def MKPipelinePass : Pass<"mk-pipeline", "mlir::ModuleOp"> {
    let summary = "Software-pipeline loops with mk.dot using SPM multi-buffering";
    let description = [{
        Reads 'tt.num_stages' from scf.for loop attributes (written by the Triton
        front-end via tl.range(..., num_stages=N)), or falls back to the global
        num-stages option.
        For each qualifying innermost scf.for that contains:
        - at least one memref.copy (DDR -> SPM load) AND
        - at least one mk.dot (compute)
        the pass rewrites the loop into a software-pipelined form with
        (num_stages - 1) prefetch buffers:
        prologue: kick off first (num_stages-1) iterations of DDR->SPM copy
        kernel:   overlap next copy with current mk.dot
        epilogue: drain remaining mk.dot after last copy
        New memref.alloc ops for the extra SPM buffers are inserted BEFORE
        spmd-allocate-shared-memory so that the existing allocator assigns
        correct allocation.offset attributes automatically.
        Barrier insertion:
        mk.barrier is inserted between copy and compute in each unrolled stage
        to synchronize SPM writes.
    }];

    let dependentDialects = ["mlir::mk::MagicKernelDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::arith::ArithDialect"];

    let options = [
    Option<"numStages", "num-stages", "unsigned", /*default=*/"2u",
           "Default software-pipeline stages when tt.num_stages loop attr is absent">,
    Option<"maxStages", "max-stages", "unsigned", /*default=*/"2u",
           "Upper clamp for effective stage count (loop attr and default)">,
    ];

}

def MKLoopBoundCanonicalizePass
    : Pass<"mk-loop-bound-canonicalize", "mlir::ModuleOp"> {
    let summary = "Canonicalize MK loop expressions using scf.for bounds";
    let description = [{
        Applies conservative loop-bound-aware rewrites after MKPipeline and
        before MK-to-Tx81 lowering. The initial pattern folds full-tile tail
        size expressions of the form:

          size = max(min(iv + step, ub), iv) - iv
          is_tail = size < step

        when the enclosing scf.for has constant bounds/step and every
        iteration is known to satisfy iv + step <= ub. This exposes constant
        copy sizes and false tail-fill guards to the following cse/canonicalize
        passes. The pass is intentionally structured so more loop-bound
        patterns can be added incrementally.
    }];

    let dependentDialects = ["mlir::scf::SCFDialect",
                             "mlir::arith::ArithDialect"];
}

#endif // MK_PIPELINE_PASSES