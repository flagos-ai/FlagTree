#ifndef TRITON_XPU_ANALYSIS_SCALAR_ANALYSIS_H
#define TRITON_XPU_ANALYSIS_SCALAR_ANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <numeric>

namespace mlir {
namespace triton {
namespace xpu {

/// Lattice element for ScalarAnalysis. Classifies an SSA Value's per-lane
/// pattern starting from `tt.make_range` / `tt.splat` along the use-def chain.
///
/// Lattice ordering:
///   Bottom (Unknown)  pessimistic / uninitialized
///   |- Scalar             uniform across lanes
///   |- VectorContig(s)    per-lane affine: base + s * lane
///   Top (VectorOther)     known not-uniform / not-affine; can't be sharpened
///
/// In addition to the kind / stride, we track a **base alignment** `baseAlign`:
///   - `baseAlign = 0`  : the value is provably exactly zero (i.e. divisible
///                        by anything; identity for gcd merges).
///   - `baseAlign = N>=1`: the per-lane base is a known multiple of `N`.
///
/// Tracking `baseAlign` lets the transfer functions for `arith.divsi` and
/// `arith.remsi` sharpen results when the divisor is compatible with the
/// known alignment (e.g. `(pid*64 + arange(64)) / 64` collapses to a Scalar).
///
/// `join` is used by the dataflow solver to merge states arriving from
/// different control-flow paths; arithmetic transfer functions live in the
/// op visitor.
struct ScalarValueState {
  enum class Kind : uint8_t {
    Unknown = 0,      // bottom
    Scalar = 1,       // lane-uniform
    VectorContig = 2, // base + stride * lane (stride known)
    BlockScalar = 3,  // uniform within each rowLen-sized block; varies between
                      // blocks (e.g. xindex / 64 when xindex = arange contig)
    BlockContig = 4,  // stride-1 contig within each rowLen-sized block; base
                      // varies between blocks (e.g. xindex % 64)
    VectorOther = 5,  // top: lane-varying and non-affine
  };

  ScalarValueState() = default;
  ScalarValueState(Kind k, int64_t s = 0, uint64_t ba = 1, int64_t rl = 0,
                   int64_t bs = 0, bool bsKnown = false, bool fromLoad = false)
      : kind(k), stride(s), baseAlign(ba), rowLen(rl), blockStride(bs),
        blockStrideKnown(bsKnown), blockFromLoad(fromLoad) {}

  static ScalarValueState scalar(uint64_t ba = 1) {
    return {Kind::Scalar, 0, ba, 0, 0, false, false};
  }
  static ScalarValueState contig(int64_t stride, uint64_t ba = 1) {
    return {Kind::VectorContig, stride, ba, 0, 0, false, false};
  }
  static ScalarValueState blockScalar(int64_t rowLen, uint64_t ba = 1,
                                      int64_t bs = 0, bool bsKnown = false,
                                      bool fromLoad = false) {
    return {Kind::BlockScalar, 0, ba, rowLen, bs, bsKnown, fromLoad};
  }
  static ScalarValueState blockContig(int64_t rowLen, uint64_t ba = 1,
                                      int64_t bs = 0, bool bsKnown = false,
                                      bool fromLoad = false) {
    return {Kind::BlockContig, 1, ba, rowLen, bs, bsKnown, fromLoad};
  }
  static ScalarValueState other() {
    return {Kind::VectorOther, 0, 1, 0, 0, false, false};
  }
  static ScalarValueState unknown() {
    return {Kind::Unknown, 0, 1, 0, 0, false, false};
  }

  bool isUnknown() const { return kind == Kind::Unknown; }
  bool isScalar() const { return kind == Kind::Scalar; }
  bool isContig() const { return kind == Kind::VectorContig; }
  bool isBlockScalar() const { return kind == Kind::BlockScalar; }
  bool isBlockContig() const { return kind == Kind::BlockContig; }
  bool isOther() const { return kind == Kind::VectorOther; }

  bool operator==(const ScalarValueState &rhs) const {
    if (kind != rhs.kind)
      return false;
    if (kind == Kind::VectorContig && stride != rhs.stride)
      return false;
    if ((kind == Kind::BlockScalar || kind == Kind::BlockContig) &&
        rowLen != rhs.rowLen)
      return false;
    if ((kind == Kind::BlockScalar || kind == Kind::BlockContig) &&
        (blockStrideKnown != rhs.blockStrideKnown ||
         (blockStrideKnown && blockStride != rhs.blockStride)))
      return false;
    if ((kind == Kind::BlockScalar || kind == Kind::BlockContig) &&
        blockFromLoad != rhs.blockFromLoad)
      return false;
    if ((kind == Kind::Scalar || kind == Kind::VectorContig ||
         kind == Kind::BlockScalar || kind == Kind::BlockContig) &&
        baseAlign != rhs.baseAlign)
      return false;
    return true;
  }

  /// Merge two alignment values. `0` is treated as "infinite alignment"
  /// (i.e. the value is exactly zero) and acts as the identity for gcd.
  static uint64_t mergeAlign(uint64_t a, uint64_t b) {
    if (a == 0)
      return b;
    if (b == 0)
      return a;
    return std::gcd(a, b);
  }

  /// Pessimistic value used by `setToEntryState` for entry/external Values
  /// (block arguments, function arguments).
  static ScalarValueState getPessimisticValueState(Value v) {
    return ScalarValueState();
  }

  /// Lattice meet: merge two states reaching the same SSA value through
  /// distinct control-flow paths.
  static ScalarValueState join(const ScalarValueState &a,
                               const ScalarValueState &b) {
    if (a.kind == Kind::Unknown)
      return b;
    if (b.kind == Kind::Unknown)
      return a;
    if (a.kind == Kind::VectorOther || b.kind == Kind::VectorOther)
      return other();
    if (a.kind == Kind::Scalar && b.kind == Kind::Scalar)
      return scalar(mergeAlign(a.baseAlign, b.baseAlign));
    if (a.kind == Kind::VectorContig && b.kind == Kind::VectorContig)
      return a.stride == b.stride
                 ? contig(a.stride, mergeAlign(a.baseAlign, b.baseAlign))
                 : other();
    if (a.kind == Kind::BlockScalar && b.kind == Kind::BlockScalar) {
      if (a.rowLen != b.rowLen)
        return other();
      bool bsk = a.blockStrideKnown && b.blockStrideKnown &&
                 a.blockStride == b.blockStride;
      return blockScalar(a.rowLen, mergeAlign(a.baseAlign, b.baseAlign),
                         bsk ? a.blockStride : 0, bsk,
                         a.blockFromLoad && b.blockFromLoad);
    }
    if (a.kind == Kind::BlockContig && b.kind == Kind::BlockContig) {
      if (a.rowLen != b.rowLen)
        return other();
      bool bsk = a.blockStrideKnown && b.blockStrideKnown &&
                 a.blockStride == b.blockStride;
      return blockContig(a.rowLen, mergeAlign(a.baseAlign, b.baseAlign),
                         bsk ? a.blockStride : 0, bsk,
                         a.blockFromLoad && b.blockFromLoad);
    }
    // Mixed kinds along the same SSA value cannot be reconciled.
    return other();
  }

  void print(raw_ostream &os) const {
    switch (kind) {
    case Kind::Unknown:
      os << "Unknown";
      break;
    case Kind::Scalar:
      os << "Scalar(baseAlign=" << baseAlign << ")";
      break;
    case Kind::VectorContig:
      os << "VectorContig(stride=" << stride << ", baseAlign=" << baseAlign
         << ")";
      break;
    case Kind::BlockScalar:
      os << "BlockScalar(rowLen=" << rowLen << ", baseAlign=" << baseAlign
         << ")";
      break;
    case Kind::BlockContig:
      os << "BlockContig(rowLen=" << rowLen << ", baseAlign=" << baseAlign
         << ")";
      break;
    case Kind::VectorOther:
      os << "VectorOther";
      break;
    }
  }

  Kind kind = Kind::Unknown;
  int64_t stride = 0;
  uint64_t baseAlign = 1;
  int64_t rowLen = 0;
  // For BlockScalar / BlockContig kinds, the inter-row stride (i.e. how the
  // block-index `idx / rowLen` is scaled before being added to the per-block
  // pattern). Only meaningful when `blockStrideKnown == true`. The canonical
  // BlockContig pattern produced by `arange % R` has blockStride == 1 (after
  // the implicit `(idx / R) * 1` from the original `arange`); patterns like
  // `(idx / R) * S + (idx % R)` carry blockStride = S.
  int64_t blockStride = 0;
  bool blockStrideKnown = false;
  // True when this Block* pattern's inter-row base is produced by a runtime
  // `triton_xpu.load` (gather), i.e. the genuine "k89" embedding-gather case
  // that LocallyContinuous with rowStride=-1 was designed for. False for
  // Block* patterns built purely from arithmetic on `arange` (e.g. the cat
  // strided-copy `(idx/R)*S + (idx%R)`), where the inter-row stride S is a
  // compile-time constant != R and must NOT be treated as row-by-row
  // contiguous DMA — those are left to OffsetAnalysis.
  bool blockFromLoad = false;
};

/// Forward sparse dataflow analysis that propagates `ScalarValueState`
/// through arithmetic / triton ops along the use-def chain.
class ScalarAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                           dataflow::Lattice<ScalarValueState>> {
public:
  using SparseForwardDataFlowAnalysis<
      dataflow::Lattice<ScalarValueState>>::SparseForwardDataFlowAnalysis;

  // LLVM 22 changed the transfer function to return LogicalResult (failure is
  // reserved for hard errors; "cannot refine" is still success).
  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::Lattice<ScalarValueState> *> operands,
      ArrayRef<dataflow::Lattice<ScalarValueState> *> results) override;

  /// Entry / external Values default to Scalar for non-tensor (a single SSA
  /// value is trivially lane-uniform) and VectorOther for tensor types
  /// (without a defining op we cannot say anything).
  void setToEntryState(dataflow::Lattice<ScalarValueState> *lattice) override {
    Value v = lattice->getAnchor();
    ScalarValueState init = isa<RankedTensorType>(v.getType())
                                ? ScalarValueState::other()
                                : ScalarValueState::scalar(1);
    propagateIfChanged(lattice, lattice->join(init));
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITON_XPU_ANALYSIS_SCALAR_ANALYSIS_H
