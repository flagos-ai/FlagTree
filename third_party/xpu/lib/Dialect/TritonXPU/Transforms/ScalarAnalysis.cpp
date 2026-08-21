//===----------------------------------------------------------------------===//
// TritonXPUScalarAnalysis pass
//
// Drives the dataflow `ScalarAnalysis` (defined in xpu/include/Analysis/
// ScalarAnalysis.h) to classify every SSA value as Scalar / VectorContig /
// VectorOther starting from `tt.make_range` and `tt.splat`. Then, for every
// `triton_xpu.gm2lm` / `triton_xpu.lm2gm` whose pointer-tensor is
// `splat(scalar_base) + Contig(stride=1)`, mark `offsetState = Continuous`
// and `handwrittenOffsetState = true` so the downstream `OffsetAnalysis`
// preserves the result.
//===----------------------------------------------------------------------===//

#include "triton/Analysis/ScalarAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "triton/Analysis/NewAnalysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <climits>
#include <cstdlib>
#include <string>

#define DEBUG_TYPE "tritonxpu-scalar-analysis"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUSCALARANALYSIS
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

struct TritonXPUScalarAnalysisPass
    : public impl::TritonXPUScalarAnalysisBase<TritonXPUScalarAnalysisPass> {

  using impl::TritonXPUScalarAnalysisBase<
      TritonXPUScalarAnalysisPass>::TritonXPUScalarAnalysisBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // 1. Run the dataflow analysis over each function body.
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<ScalarAnalysis>();
    if (failed(solver.initializeAndRun(mod))) {
      signalPassFailure();
      return;
    }

    // 2. Walk gm2lm / lm2gm and stamp Continuous + handwritten when the
    //    pointer's classification is exactly VectorContig(stride = 1).
    auto isContig1 = [&](Value ptrTensor) -> bool {
      auto *lattice =
          solver.lookupState<dataflow::Lattice<ScalarValueState>>(ptrTensor);
      if (!lattice)
        return false;
      const ScalarValueState &v = lattice->getValue();
      return v.isContig() && v.stride == 1;
    };

    // Returns rowLen (>= 2) when the pointer-tensor is BlockContig with
    // stride=1 AND it is a *genuine runtime gather* (its inter-row base comes
    // from a `triton_xpu.load`, the k89 embedding case) AND it is *not
    // provably unsafe* to mark as LocallyContinuous.
    //
    // The downstream lowering treats LocallyContinuous(rowLen=R, rowStride=-1)
    // as "iterate R-lane blocks in linear order, row base is data-dependent".
    // That is correct only for true gathers, where each R-lane row is still
    // walked in element order and the inter-row jump is a runtime value.
    //
    // It is NOT correct for Block* patterns built purely from arithmetic on
    // `arange`, e.g. the cat strided-copy offset `(idx/R)*S + (idx%R)` where
    // the inter-row stride S is a compile-time constant != R. Such patterns
    // can surface as BlockContig with an *unknown* blockStride after the
    // multi-dim index math is combined, which would slip past a "blockStride
    // known && != R" check. Requiring `blockFromLoad` excludes them: they are
    // left to OffsetAnalysis, which computes the correct fixed rowStride.
    //
    // blockFromLoad: true when the BlockContig's inter-row base originates
    // from a `triton_xpu.load` result (i.e., the row base address is truly
    // data-dependent at runtime, not derivable from arithmetic on arange).
    //
    // In aggressive mode (second run after OffsetAnalysis), the blockFromLoad
    // guard is skipped: if OffsetAnalysis already failed (offsetState=-1),
    // we accept arithmetic-derived BlockContig and mark LocallyContinuous as
    // a best-effort fallback.
    //
    // We still reject when blockStride is statically known and != rowLen
    // (e.g. flip's -R) as an extra guard.
    auto getBlockContigRowLen = [&](Value ptrTensor) -> int64_t {
      auto *lattice =
          solver.lookupState<dataflow::Lattice<ScalarValueState>>(ptrTensor);
      if (!lattice)
        return 0;
      const ScalarValueState &v = lattice->getValue();
      if (!v.isBlockContig() || v.stride != 1 || v.rowLen < 2)
        return 0;
      if (!aggressive && !v.blockFromLoad)
        return 0; // not a genuine runtime gather (e.g. cat strided copy)
      if (v.blockStrideKnown && v.blockStride != v.rowLen)
        return 0; // provably unsafe (e.g. flip's -R)
      return v.rowLen;
    };

    auto markContinuous = [&](Operation *op) {
      OpBuilder b(op);
      op->setAttr("offsetState", b.getSI32IntegerAttr(static_cast<int32_t>(
                                     OffsetState::Continuous)));
      op->setAttr("handwrittenOffsetState", b.getBoolAttr(true));
      // Stamp the auxiliary attributes that downstream OffsetAnalysis would
      // otherwise compute via the (now skipped) inference path. For the
      // VectorContig(stride=1) case these values are statically known:
      //   fixedStride = 1 (contiguous element stride)
      //   lrie        = 1 (no runs of identical elements)
      // rowLen / rowStride / tensorColSize remain -1 (only meaningful for
      // LocallyContinuous / multi-row patterns), matching what
      // `getOffsetState` produces for plain Continuous.
      if (isa<triton::xpu::GM2LMOp>(op)) {
        op->setAttr("fixedStride", b.getSI32IntegerAttr(1));
        op->setAttr("lrie", b.getSI32IntegerAttr(1));
      }
    };

    // Stamp LocallyContinuous with the detected rowLen. rowStride is left
    // at -1 because the inter-row stride is data-dependent (e.g. comes from
    // a gather index). The downstream LLVM lowering's LocallyContinuous path
    // handles rowStride==-1 (row-by-row DMA) and even upgrades to Continuous
    // when rowLen % numElems == 0.
    auto markLocallyContinuous = [&](Operation *op, int64_t rowLen) {
      OpBuilder b(op);
      op->setAttr("offsetState", b.getSI32IntegerAttr(static_cast<int32_t>(
                                     OffsetState::LocallyContinuous)));
      op->setAttr("handwrittenOffsetState", b.getBoolAttr(true));
      op->setAttr(
          "rowLen",
          b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/true), rowLen));
      op->setAttr(
          "rowStride",
          b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/true), -1));
    };

    // Returns rowLen (>= sizePerCore) when the pointer-tensor is BlockScalar,
    // meaning every `rowLen` lanes share the same address. Only mark when
    // rowLen >= sizePerCore so that within each core the address is uniform.
    auto getBlockScalarRowLen = [&](Value ptrTensor) -> int64_t {
      auto *lattice =
          solver.lookupState<dataflow::Lattice<ScalarValueState>>(ptrTensor);
      if (!lattice)
        return 0;
      const ScalarValueState &v = lattice->getValue();
      if (!v.isBlockScalar() || v.rowLen < 2)
        return 0;
      auto ty = dyn_cast<RankedTensorType>(ptrTensor.getType());
      if (!ty)
        return 0;
      auto enc = dyn_cast<triton::xpu::ClusterLayoutAttr>(ty.getEncoding());
      if (!enc)
        return 0;
      int64_t sizePerCore = enc.getSizePerCore()[0];
      if (sizePerCore <= 0 || v.rowLen % sizePerCore != 0)
        return 0;
      return v.rowLen;
    };

    auto markDiscreteSame = [&](Operation *op, int64_t /*rowLen*/) {
      OpBuilder b(op);
      op->setAttr("offsetState", b.getSI32IntegerAttr(static_cast<int32_t>(
                                     OffsetState::DiscreteSame)));
      op->setAttr("handwrittenOffsetState", b.getBoolAttr(true));
      op->setAttr("fixedStride", b.getSI32IntegerAttr(0));
      op->setAttr("rowLen", b.getIntegerAttr(
                                b.getIntegerType(64, /*isSigned=*/true), -1));
      op->setAttr(
          "rowStride",
          b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/true), -1));
    };

    // Returns rowLen when the pointer-tensor is BlockScalar but each core spans
    // *multiple whole blocks* (rowLen < sizePerCore && sizePerCore % rowLen ==
    // 0). DiscreteSame requires one uniform address per core (rowLen >=
    // sizePerCore); this complementary "locally scalar" case keeps the
    // per-block scalar address structure so the gm2lm can issue one scalar DMA
    // per block instead of falling back to the per-element Unknown gather.
    auto getLocalScalarRowLen = [&](Value ptrTensor) -> int64_t {
      auto *lattice =
          solver.lookupState<dataflow::Lattice<ScalarValueState>>(ptrTensor);
      if (!lattice)
        return 0;
      const ScalarValueState &v = lattice->getValue();
      if (!v.isBlockScalar() || v.rowLen < 2)
        return 0;
      auto ty = dyn_cast<RankedTensorType>(ptrTensor.getType());
      if (!ty)
        return 0;
      auto enc = dyn_cast<triton::xpu::ClusterLayoutAttr>(ty.getEncoding());
      if (!enc)
        return 0;
      int64_t sizePerCore = enc.getSizePerCore()[0];
      if (sizePerCore <= 0)
        return 0;
      if (v.rowLen >= sizePerCore)
        return 0; // DiscreteSame territory (handled by getBlockScalarRowLen)
      if (sizePerCore % v.rowLen != 0)
        return 0; // need whole blocks within a core
      return v.rowLen;
    };

    auto markLocallyScalar = [&](Operation *op, int64_t rowLen) {
      OpBuilder b(op);
      op->setAttr("offsetState", b.getSI32IntegerAttr(static_cast<int32_t>(
                                     OffsetState::LocallyScalar)));
      op->setAttr("handwrittenOffsetState", b.getBoolAttr(true));
      // INT32_MIN keeps the consuming load off the DiscreteSame(stride==0)
      // path; OffsetAnalysis stamps isDiscrete=true so it uses the per-lane LM
      // read.
      op->setAttr("fixedStride", b.getSI32IntegerAttr(INT32_MIN));
      op->setAttr(
          "rowLen",
          b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/true), rowLen));
      op->setAttr(
          "rowStride",
          b.getIntegerAttr(b.getIntegerType(64, /*isSigned=*/true), -1));
    };

    // In aggressive mode, only update ops where offsetState is still -1
    // (i.e., OffsetAnalysis failed to determine the state).
    auto shouldSkipInAggressive = [&](Operation *op) -> bool {
      if (!aggressive)
        return false;
      auto attr = op->getAttrOfType<IntegerAttr>("offsetState");
      if (!attr)
        return false; // no attr means not yet processed, ok to mark
      return attr.getValue().getSExtValue() !=
             -1; // already has valid state, skip
    };

    mod.walk([&](Operation *op) {
      if (auto gm2lm = dyn_cast<triton::xpu::GM2LMOp>(op)) {
        if (gm2lm.getHandwrittenOffsetState())
          return;
        if (shouldSkipInAggressive(op))
          return;
        if (isContig1(gm2lm.getPtr())) {
          markContinuous(op);
          return;
        }
        if (int64_t rl = getBlockContigRowLen(gm2lm.getPtr())) {
          markLocallyContinuous(op, rl);
          return;
        }
        if (int64_t rl = getBlockScalarRowLen(gm2lm.getPtr())) {
          markDiscreteSame(op, rl);
          return;
        }
        if (int64_t rl = getLocalScalarRowLen(gm2lm.getPtr())) {
          markLocallyScalar(op, rl);
          return;
        }
        return;
      }
      if (auto lm2gm = dyn_cast<triton::xpu::LM2GMOp>(op)) {
        if (lm2gm.getHandwrittenOffsetState())
          return;
        if (shouldSkipInAggressive(op))
          return;
        if (isContig1(lm2gm.getPtr())) {
          markContinuous(op);
          return;
        }
        // dont mark locally continuous or DiscreteSame for lm2gm
        // if (int64_t rl = getBlockContigRowLen(lm2gm.getPtr())) {
        //   markLocallyContinuous(op, rl);
        //   return;
        // }
        // if (int64_t rl = getBlockScalarRowLen(lm2gm.getPtr())) {
        //   markDiscreteSame(op, rl);
        // }
        return;
      }
    });

    // 3. Debug mode: stamp every tensor result with its lattice state.
    //    Enable via env TRITONXPU_SCALAR_ANALYSIS_DEBUG=1
    if (std::getenv("TRITONXPU_SCALAR_ANALYSIS_DEBUG")) {
      mod.walk([&](Operation *op) {
        for (Value res : op->getResults()) {
          auto *lattice =
              solver.lookupState<dataflow::Lattice<ScalarValueState>>(res);
          if (!lattice)
            continue;
          const ScalarValueState &v = lattice->getValue();
          std::string desc;
          llvm::raw_string_ostream os(desc);
          v.print(os);
          OpBuilder b(op);
          op->setAttr("scalar_state", b.getStringAttr(desc));
        }
      });
    }
  }
};

} // namespace
} // namespace xpu
} // namespace triton
} // namespace mlir
