#include "triton/Analysis/ScalarAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::dataflow;

#define DEBUG_TYPE "tritonxpu-scalar-analysis"

namespace mlir {
namespace triton {
namespace xpu {

namespace {

using State = ScalarValueState;

static bool isTensor(Value v) { return isa<RankedTensorType>(v.getType()); }

/// Default lattice value when an SSA value is produced by an op we don't
/// recognise — depend on whether the result is a tensor (lane-shaped) or a
/// plain scalar.
static State defaultFor(Value v) {
  return isTensor(v) ? State::other() : State::scalar(1);
}

/// Try to extract the (signed) integer value of a constant scalar / splat.
static std::optional<int64_t> getConstInt(Value v) {
  if (!v)
    return std::nullopt;
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto cst = dyn_cast<arith::ConstantOp>(def);
  if (!cst)
    return std::nullopt;
  Attribute attr = cst.getValue();
  if (auto ia = dyn_cast<IntegerAttr>(attr))
    return ia.getValue().getSExtValue();
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (dense.isSplat())
      return dense.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

/// Alignment of a known integer constant (0 → exactly zero / "infinite").
static uint64_t alignOfConst(int64_t c) {
  if (c == 0)
    return 0;
  uint64_t u = c < 0 ? static_cast<uint64_t>(-c) : static_cast<uint64_t>(c);
  return u;
}

/// Saturating multiply of two alignment values, treating 0 as identity-zero
/// (i.e. exactly-zero values stay exactly-zero under multiplication).
static uint64_t mulAlign(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0)
    return 0;
  // Avoid overflow; cap at a large power of two.
  if (a > (uint64_t(1) << 32) || b > (uint64_t(1) << 32))
    return 1;
  return a * b;
}

/// Lane count of a tensor result, or 0 if not a ranked tensor.
static int64_t laneCount(Value v) {
  auto rt = dyn_cast<RankedTensorType>(v.getType());
  if (!rt)
    return 0;
  int64_t n = 1;
  for (int64_t d : rt.getShape())
    n *= d;
  return n;
}

/// `arith.addi` / `arith.subi` transfer.
static State combineAdd(const State &a, const State &b, bool isSub) {
  if (a.isUnknown() || b.isUnknown())
    return State::unknown();
  if (a.isOther() || b.isOther())
    return State::other();
  uint64_t ba = State::mergeAlign(a.baseAlign, b.baseAlign);
  if (a.isScalar() && b.isScalar())
    return State::scalar(ba);
  if (a.isScalar() && b.isContig())
    return State::contig(isSub ? -b.stride : b.stride, ba);
  if (b.isScalar() && a.isContig())
    return State::contig(a.stride, ba);
  // Scalar + BlockX  -> BlockX (Scalar is lane-uniform, doesn't disturb the
  // per-block pattern; blockStride is preserved).
  if (a.isScalar() && (b.isBlockScalar() || b.isBlockContig())) {
    int64_t bs = b.blockStride;
    if (isSub) bs = -bs;
    if (b.isBlockScalar())
      return State::blockScalar(b.rowLen, ba, bs, b.blockStrideKnown,
                                b.blockFromLoad);
    if (isSub)
      return State::other();
    return State::blockContig(b.rowLen, ba, bs, b.blockStrideKnown,
                              b.blockFromLoad);
  }
  if (b.isScalar() && (a.isBlockScalar() || a.isBlockContig())) {
    if (a.isBlockScalar())
      return State::blockScalar(a.rowLen, ba, a.blockStride,
                                a.blockStrideKnown, a.blockFromLoad);
    return State::blockContig(a.rowLen, ba, a.blockStride,
                              a.blockStrideKnown, a.blockFromLoad);
  }
  // BlockScalar + BlockContig with same rowLen -> BlockContig.
  // The resulting blockStride is the sum (or difference) of the two
  // blockStrides. BlockContig's blockStride is "0" for the canonical
  // `arange % R` pattern; BlockScalar's blockStride is "S" for
  // `(arange / R) * S`.
  // When subtracting a BlockContig, the intra-block stride flips sign
  // (e.g. `S - (idx%R)` has stride=-1 within each R-block).
  if (a.isBlockScalar() && b.isBlockContig() && a.rowLen == b.rowLen) {
    if (isSub)
      return State::other();
    bool bsk = a.blockStrideKnown && b.blockStrideKnown;
    int64_t bs = bsk ? (a.blockStride + b.blockStride) : 0;
    return State::blockContig(a.rowLen, ba, bs, bsk,
                              a.blockFromLoad || b.blockFromLoad);
  }
  if (b.isBlockScalar() && a.isBlockContig() && a.rowLen == b.rowLen) {
    if (isSub)
      return State::other();
    bool bsk = a.blockStrideKnown && b.blockStrideKnown;
    int64_t bs = bsk ? (a.blockStride + b.blockStride) : 0;
    return State::blockContig(a.rowLen, ba, bs, bsk,
                              a.blockFromLoad || b.blockFromLoad);
  }
  // BlockContig(R1) + BlockScalar(R2) where R1 | R2: BlockScalar is uniform
  // within each R1-block (since R1 divides R2), so contig pattern preserved.
  if (a.isBlockContig() && b.isBlockScalar() && a.rowLen != b.rowLen) {
    int64_t R1 = a.rowLen, R2 = b.rowLen;
    if (R1 > 0 && R2 > 0 && (R2 % R1) == 0) {
      return State::blockContig(R1, ba, 0, false,
                                a.blockFromLoad || b.blockFromLoad);
    }
  }
  if (b.isBlockContig() && a.isBlockScalar() && a.rowLen != b.rowLen) {
    int64_t R1 = b.rowLen, R2 = a.rowLen;
    if (R1 > 0 && R2 > 0 && (R2 % R1) == 0) {
      if (isSub)
        return State::other();
      return State::blockContig(R1, ba, 0, false,
                                a.blockFromLoad || b.blockFromLoad);
    }
  }
  // BlockScalar + BlockScalar with same rowLen -> BlockScalar.
  if (a.isBlockScalar() && b.isBlockScalar() && a.rowLen == b.rowLen) {
    bool bsk = a.blockStrideKnown && b.blockStrideKnown;
    int64_t bs = bsk ? (isSub ? a.blockStride - b.blockStride
                              : a.blockStride + b.blockStride)
                     : 0;
    return State::blockScalar(a.rowLen, ba, bs, bsk,
                              a.blockFromLoad || b.blockFromLoad);
  }
  // BlockScalar(R1) + BlockScalar(R2) where R1 | R2 or R2 | R1:
  // uniform within the smaller block is preserved.
  if (a.isBlockScalar() && b.isBlockScalar() && a.rowLen != b.rowLen) {
    int64_t R1 = a.rowLen, R2 = b.rowLen;
    if (R1 > 0 && R2 > 0) {
      int64_t Rmin = std::min(R1, R2);
      int64_t Rmax = std::max(R1, R2);
      if ((Rmax % Rmin) == 0) {
        return State::blockScalar(Rmin, ba, 0, false,
                                  a.blockFromLoad || b.blockFromLoad);
      }
    }
  }
  // BlockContig + BlockContig with same rowLen: stride doubles -> not contig-1
  // anymore; degrade.
  if (a.isBlockContig() && b.isBlockContig())
    return State::other();
  if (a.isContig() && b.isContig()) {
    int64_t s = isSub ? a.stride - b.stride : a.stride + b.stride;
    return State::contig(s, ba);
  }
  // Mixed (e.g. VectorContig + BlockX) — conservative.
  return State::other();
}

/// `arith.muli` transfer. Sharpens with a known constant operand when
/// possible:
///   Scalar(ba) * const(c)   -> Scalar(ba * |c|)
///   VC(s, ba) * const(c)    -> VC(s*c, ba * |c|)   (tensor result)
static State combineMul(Value lhs, Value rhs, const State &a, const State &b,
                        bool resultIsTensor) {
  if (a.isUnknown() || b.isUnknown())
    return State::unknown();
  if (a.isOther() || b.isOther())
    return resultIsTensor ? State::other() : State::scalar(1);

  std::optional<int64_t> cl = getConstInt(lhs);
  std::optional<int64_t> cr = getConstInt(rhs);

  // Scalar * Scalar -> Scalar
  if (a.isScalar() && b.isScalar()) {
    uint64_t ba = mulAlign(a.baseAlign, b.baseAlign);
    return State::scalar(ba);
  }
  // VC * Scalar with known constant on the scalar side -> sharpen stride.
  if (a.isContig() && b.isScalar() && cr) {
    int64_t c = *cr;
    return State::contig(a.stride * c, mulAlign(a.baseAlign, alignOfConst(c)));
  }
  if (b.isContig() && a.isScalar() && cl) {
    int64_t c = *cl;
    return State::contig(b.stride * c, mulAlign(b.baseAlign, alignOfConst(c)));
  }
  // VC * Scalar without a known constant: stride unknown -> conservative.
  // BlockScalar * Scalar (or const) -> BlockScalar (rowLen preserved).
  // If the scalar side is a known constant, scale blockStride by that const.
  if (a.isBlockScalar() && b.isScalar()) {
    uint64_t ba = cr ? mulAlign(a.baseAlign, alignOfConst(*cr))
                     : mulAlign(a.baseAlign, b.baseAlign);
    if (cr && a.blockStrideKnown) {
      return State::blockScalar(a.rowLen, ba, a.blockStride * (*cr), true,
                                a.blockFromLoad);
    }
    return State::blockScalar(a.rowLen, ba, 0, false, a.blockFromLoad);
  }
  if (b.isBlockScalar() && a.isScalar()) {
    uint64_t ba = cl ? mulAlign(b.baseAlign, alignOfConst(*cl))
                     : mulAlign(b.baseAlign, a.baseAlign);
    if (cl && b.blockStrideKnown) {
      return State::blockScalar(b.rowLen, ba, b.blockStride * (*cl), true,
                                b.blockFromLoad);
    }
    return State::blockScalar(b.rowLen, ba, 0, false, b.blockFromLoad);
  }
  // BlockContig * const: stride changes from 1 to c -> no longer contig-1.
  // Degrade unless c == 1.
  if (a.isBlockContig() && b.isScalar()) {
    if (cr && *cr == 1)
      return a;
    return resultIsTensor ? State::other() : State::scalar(1);
  }
  if (b.isBlockContig() && a.isScalar()) {
    if (cl && *cl == 1)
      return b;
    return resultIsTensor ? State::other() : State::scalar(1);
  }
  return resultIsTensor ? State::other() : State::scalar(1);
}

/// `arith.remsi` transfer with a constant divisor `c`.
///   Scalar(ba) % c            -> Scalar(gcd(ba, c))
///   VC(s, ba) % c, where the lane span s*(n-1) < gcd(ba, c)
///                              -> VC(s, gcd(ba, c))
/// Otherwise conservative.
static State remsiByConst(const State &lhs, int64_t c, int64_t lanes,
                          bool resultIsTensor) {
  if (c == 0)
    return resultIsTensor ? State::other() : State::scalar(1);
  uint64_t cu = static_cast<uint64_t>(c < 0 ? -c : c);
  if (lhs.isScalar()) {
    uint64_t ba = State::mergeAlign(lhs.baseAlign, cu);
    return State::scalar(ba);
  }
  if (lhs.isContig()) {
    uint64_t ba = State::mergeAlign(lhs.baseAlign, cu);
    int64_t span = lhs.stride * (lanes > 0 ? lanes - 1 : 0);
    int64_t absSpan = span < 0 ? -span : span;
    // No-wrap condition: every lane stays within a single c-block.
    if (lhs.baseAlign != 0 && (cu % lhs.baseAlign) == 0 &&
        static_cast<uint64_t>(absSpan) < lhs.baseAlign) {
      return State::contig(lhs.stride, ba);
    }
    if (lhs.baseAlign == 0 && static_cast<uint64_t>(absSpan) < cu) {
      // Base is exactly zero; full span fits in [0, c).
      return State::contig(lhs.stride, ba);
    }
    // Wraps: VC(1, ba) % c with span >= c. Within each c-lane chunk the
    // pattern is contig-1 (possibly shifted by base%c). Promote to
    // BlockContig(rowLen=c). Used by patterns like `xindex % 576` when
    // followed by further `% 64` -- as long as 64 | 576 the wraps align
    // with the 64-lane block boundaries.
    // The canonical `arange % R` carries blockStride = 0 (no inter-row term).
    if (lhs.stride == 1) {
      return State::blockContig(c < 0 ? -c : c, ba, 0, true);
    }
  }
  // BlockContig(rowLen=R, stride=1) % c with c | R -> BlockContig(rowLen=c).
  if (lhs.isBlockContig() && lhs.stride == 1) {
    int64_t R = lhs.rowLen;
    if (R > 0 && c > 0) {
      if ((R % c) == 0) {
        uint64_t ba = State::mergeAlign(lhs.baseAlign, cu);
        return State::blockContig(c, ba, 0, true);
      }
      // c >= R: within each R-block lanes are 0..R-1, all < c -> unchanged.
      if ((c % R) == 0) {
        uint64_t ba = State::mergeAlign(lhs.baseAlign, cu);
        return State::blockContig(R, ba, lhs.blockStride, lhs.blockStrideKnown,
                                  lhs.blockFromLoad);
      }
    }
  }
  // BlockScalar % c -> BlockScalar (uniform within block stays uniform).
  if (lhs.isBlockScalar()) {
    uint64_t ba = State::mergeAlign(lhs.baseAlign, cu);
    return State::blockScalar(lhs.rowLen, ba, 0, false, lhs.blockFromLoad);
  }
  return resultIsTensor ? State::other() : State::scalar(1);
}

/// `arith.divsi` transfer with a constant divisor `c`.
///   Scalar(ba) / c, where c | ba   -> Scalar(ba / c)
///   VC(s, ba) / c, where c | ba and stride*(n-1) < c
///                                  -> Scalar(ba / c)   (all lanes equal)
/// Otherwise conservative.
static State divsiByConst(const State &lhs, int64_t c, int64_t lanes,
                          bool resultIsTensor) {
  if (c == 0)
    return resultIsTensor ? State::other() : State::scalar(1);
  uint64_t cu = static_cast<uint64_t>(c < 0 ? -c : c);
  if (lhs.isScalar()) {
    if (lhs.baseAlign == 0)
      return State::scalar(0);
    if ((lhs.baseAlign % cu) == 0)
      return State::scalar(lhs.baseAlign / cu);
    return State::scalar(1);
  }
  if (lhs.isContig()) {
    int64_t span = lhs.stride * (lanes > 0 ? lanes - 1 : 0);
    int64_t absSpan = span < 0 ? -span : span;
    uint64_t abs_u = static_cast<uint64_t>(absSpan);
    bool baseIsZero = (lhs.baseAlign == 0);
    bool baseDividesByC =
        (lhs.baseAlign != 0) && ((lhs.baseAlign % cu) == 0);   // c | B
    bool cDividesByBase =
        (lhs.baseAlign != 0) && ((cu % lhs.baseAlign) == 0);   // B | c
    // Case 2a: v[0] on a c-boundary (v[0]=0 or c | B), and absSpan < c
    // → all lanes fall in the same c-bucket → Scalar.
    // Quotient k = v[0]/c = m·(B/c), so result baseAlign = B/c
    // (or 0 when baseIsZero, i.e. quotient is exactly zero).
    //
    // Threshold must be c (not baseAlign): when B > c, a baseAlign-window
    // straddles B/c c-buckets, so "no wrap across baseAlign" does NOT imply
    // "no wrap across c".
    //
    // Requires stride >= 0: with negative stride the segment extends below
    // v[0], and being on a c-boundary puts v[0] at the top of the previous
    // bucket, so the segment immediately leaves it.
    //
    // Example: `(pid_y*8192 + arange(0, 64)) // 128`
    //     B=8192, c=128, k = m·(B/c) = m·64
    //     → Scalar(baseAlign=64)
    if ((baseIsZero || baseDividesByC) && lhs.stride >= 0 && abs_u < cu) {
      uint64_t ba = baseIsZero ? 0 : (lhs.baseAlign / cu);
      return State::scalar(ba);
    }
    // Case 2b: B | c (c is an integer multiple of B). v[0] lies on the
    // B-grid but may sit at the tail of a c-bucket. When absSpan < B, the
    // segment cannot reach the next B-grid point → stays in one c-bucket
    // → Scalar. Quotient k = m ÷ (c/B) (integer division) can take any
    // integer value → result baseAlign = 1.
    //
    // Threshold must be B (not c): v[0] may lie only B away from the next
    // c-boundary, so absSpan reaching B risks crossing it. Dual to Case 2a
    // where v[0] is guaranteed at a c-bucket start and can span a full c.
    //
    // Example: `(pid*1024 + arange(0, 1024)) // 409600`
    //     B=1024, c=400·B, k = m÷(c/B) = m÷400 (integer division)
    //     → Scalar(baseAlign=1)
    if (!baseIsZero && cDividesByBase && lhs.stride >= 0 &&
        abs_u < lhs.baseAlign) {
      return State::scalar(1);
    }
    // Wraps: VC(1, ba) / c with span >= c. Across c-lane chunks the
    // quotient increases by 1; within each c-lane chunk it's uniform.
    // -> BlockScalar(rowLen=c) with blockStride = 1.
    if (lhs.stride == 1) {
      uint64_t ba = baseIsZero ? 0 : 1;
      return State::blockScalar(c < 0 ? -c : c, ba, 1, true);
    }
  }
  // BlockScalar / c: uniform within block stays uniform -> BlockScalar.
  if (lhs.isBlockScalar()) {
    return State::blockScalar(lhs.rowLen, 1, 0, false, lhs.blockFromLoad);
  }
  // BlockContig(rowLen=R, stride=1) / c with c | R -> BlockScalar(rowLen=c).
  // Within each R-block lanes are 0..R-1; dividing by c gives (R/c) groups of
  // c equal quotients per R-block -> uniform within c-blocks.
  if (lhs.isBlockContig() && lhs.stride == 1) {
    int64_t R = lhs.rowLen;
    if (R > 0 && c > 0 && (R % c) == 0) {
      return State::blockScalar(c, 1);
    }
    // R | c: within each R-block span = R-1 < c, so all lanes in a block
    // share the same quotient -> BlockScalar(rowLen=R).
    if (R > 0 && c > 0 && (c % R) == 0) {
      return State::blockScalar(R, 1, 0, false, lhs.blockFromLoad);
    }
  }
  return resultIsTensor ? State::other() : State::scalar(1);
}

} // namespace

LogicalResult ScalarAnalysis::visitOperation(
    Operation *op,
    ArrayRef<const Lattice<ScalarValueState> *> operands,
    ArrayRef<Lattice<ScalarValueState> *> results) {

  auto setResult = [&](unsigned i, const State &st) {
    propagateIfChanged(results[i], results[i]->join(st));
  };

  // tt.make_range : VectorContig(stride = 1, baseAlign = |start|).
  if (auto mr = dyn_cast<triton::MakeRangeOp>(op)) {
    int64_t start = static_cast<int64_t>(mr.getStart());
    setResult(0, State::contig(1, alignOfConst(start)));
    return success();
  }

  // tt.splat : Scalar carrying the operand's alignment.
  if (isa<triton::SplatOp>(op)) {
    State a = operands[0]->getValue();
    if (a.isUnknown()) {
      setResult(0, State::unknown());
    } else if (a.isScalar()) {
      setResult(0, State::scalar(a.baseAlign));
    } else {
      // Splatting a value that itself was tracked as a vector shouldn't
      // happen in practice; be conservative.
      setResult(0, State::scalar(1));
    }
    return success();
  }

  // arith.constant : splat dense / scalar constant ⇒ Scalar(alignOfConst);
  // non-splat dense tensor ⇒ VectorOther.
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    Value res = cst.getResult();
    Attribute attr = cst.getValue();
    if (!isTensor(res)) {
      if (auto ia = dyn_cast<IntegerAttr>(attr)) {
        setResult(0, State::scalar(alignOfConst(ia.getValue().getSExtValue())));
      } else {
        setResult(0, State::scalar(1));
      }
    } else if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
      if (dense.isSplat()) {
        if (auto di = dyn_cast<DenseIntElementsAttr>(dense)) {
          setResult(0, State::scalar(alignOfConst(
                           di.getSplatValue<APInt>().getSExtValue())));
        } else {
          setResult(0, State::scalar(1));
        }
      } else {
        setResult(0, State::other());
      }
    } else {
      setResult(0, State::other());
    }
    return success();
  }

  // arith.extsi / arith.extui / arith.trunci : integer width cast does NOT
  // change the lane pattern (Scalar / Contig / BlockX all preserved).
  // Without this rule, expressions like
  //   xindex = pid.to(i64) * XBLOCK + arange(0, XBLOCK).to(i64)
  // would degrade to VectorOther at the extsi on `arange`, breaking the
  // entire downstream BlockContig / Continuous detection.
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op)) {
    setResult(0, operands[0]->getValue());
    return success();
  }

  // arith.addi / arith.subi.
  if (isa<arith::AddIOp, arith::SubIOp>(op)) {
    State a = operands[0]->getValue();
    State b = operands[1]->getValue();
    setResult(0, combineAdd(a, b, isa<arith::SubIOp>(op)));
    return success();
  }

  // arith.muli.
  if (isa<arith::MulIOp>(op)) {
    State a = operands[0]->getValue();
    State b = operands[1]->getValue();
    setResult(0, combineMul(op->getOperand(0), op->getOperand(1), a, b,
                            isTensor(op->getResult(0))));
    return success();
  }

  // arith.remsi by a known constant divisor.
  if (isa<arith::RemSIOp, arith::RemUIOp>(op)) {
    State a = operands[0]->getValue();
    auto rhs = getConstInt(op->getOperand(1));
    if (!rhs) {
      // Non-constant divisor: BlockScalar % anything is still BlockScalar
      // (uniform within each rowLen-lane block stays uniform). All other
      // kinds degrade conservatively.
      if (a.isBlockScalar()) {
        setResult(0, State::blockScalar(a.rowLen, 1, 0, false,
                                        a.blockFromLoad));
      } else {
        setResult(0, defaultFor(op->getResult(0)));
      }
      return success();
    }
    setResult(0, remsiByConst(a, *rhs, laneCount(op->getResult(0)),
                              isTensor(op->getResult(0))));
    return success();
  }

  // arith.divsi / divui by a known constant divisor.
  if (isa<arith::DivSIOp, arith::DivUIOp>(op)) {
    State a = operands[0]->getValue();
    auto rhs = getConstInt(op->getOperand(1));
    if (!rhs) {
      // Non-constant divisor: BlockScalar / anything is still BlockScalar.
      if (a.isBlockScalar()) {
        setResult(0, State::blockScalar(a.rowLen, 1, 0, false,
                                        a.blockFromLoad));
      } else {
        setResult(0, defaultFor(op->getResult(0)));
      }
      return success();
    }
    setResult(0, divsiByConst(a, *rhs, laneCount(op->getResult(0)),
                              isTensor(op->getResult(0))));
    return success();
  }

  // arith.select : if both branches share the same lattice kind, the
  // selected value retains that kind (with merged alignments). Otherwise
  // degrade conservatively. Note: this assumes the condition itself does
  // not need to be lane-uniform — we only care about the value lattice.
  if (auto sel = dyn_cast<arith::SelectOp>(op)) {
    State t = operands[1]->getValue();
    State f = operands[2]->getValue();
    setResult(0, State::join(t, f));
    return success();
  }

  // tt.addptr(ptr_tensor, offset_tensor) -> ptr_tensor.
  if (auto addPtr = dyn_cast<triton::AddPtrOp>(op)) {
    State base = operands[0]->getValue();
    State off = operands[1]->getValue();
    if (base.isUnknown() || off.isUnknown()) {
      setResult(0, State::unknown());
    } else if (base.isOther() || off.isOther()) {
      setResult(0, State::other());
    } else if (base.isScalar() && off.isScalar()) {
      setResult(0, State::scalar(1));
    } else if (base.isScalar()) {
      // Pointer base lane-uniform; per-lane offset dictates the pattern.
      setResult(0, off);
    } else if (off.isScalar()) {
      setResult(0, base);
    } else if (base.isContig() && off.isContig()) {
      // Both VectorContig.
      setResult(0, State::contig(base.stride + off.stride, 1));
    } else {
      // Mixed including BlockX / VectorContig — too complex to track precisely.
      setResult(0, State::other());
    }
    return success();
  }

  // triton_xpu.gm2lm : transparent w.r.t. the pointer's lane pattern. The
  // output is a pointer-tensor pointing into LM that mirrors the GM input.
  if (isa<triton::xpu::GM2LMOp>(op)) {
    setResult(0, operands[0]->getValue());
    return success();
  }

  // triton_xpu.load : if the pointer is lane-uniform (Scalar), the loaded
  // value is also lane-uniform across active lanes. We don't know the
  // value's alignment, so use baseAlign = 1.
  if (isa<triton::xpu::LoadOp>(op)) {
    State ptr = operands[0]->getValue();
    if (ptr.isScalar()) {
      setResult(0, State::scalar(1));
      return success();
    }
    // BlockScalar pointer -> loaded value uniform within each block.
    if (ptr.isBlockScalar()) {
      // Mark `blockFromLoad`: the per-block value is produced by a runtime
      // gather. This is the genuine k89 embedding-gather case that the
      // LocallyContinuous(rowStride=-1) row-by-row DMA path was designed for.
      setResult(0, State::blockScalar(ptr.rowLen, 1, 0, false,
                                      /*fromLoad=*/true));
      return success();
    }
    setResult(0, defaultFor(op->getResult(0)));
    return success();
  }

  // Default: produce VectorOther for tensor results, Scalar for non-tensor.
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    setResult(i, defaultFor(op->getResult(i)));
  }
  return success();
}

} // namespace xpu
} // namespace triton
} // namespace mlir
