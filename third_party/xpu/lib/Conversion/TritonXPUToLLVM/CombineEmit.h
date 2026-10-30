#ifndef TRITONXPU_CONVERSION_COMBINE_EMIT_H
#define TRITONXPU_CONVERSION_COMBINE_EMIT_H

//===----------------------------------------------------------------------===//
// Combine-region emission shared by the reduce and scan lowerings.
//
// One combine step is emitted from the region itself (op by op) instead of
// inlining the region: the same region has to run on vector<NxT> values
// (within-core stage) and on scalars (across-core stage), and a cloned region
// op carries a fixed result type, so inlining mixes operand and result widths
// once the region is retyped to vector form. These helpers stamp each emitted
// op with its actual operand types.
//
// `canInterpretCombine` is the all-or-nothing admission: everything it
// accepts must have an arm in `emitCombineOp`. The admission list has to stay
// in sync with REDUCE_COMBINE_REGION_OP / COMBINE_OP in
// xpu/include/Analysis/VectorizabilityAnalysis.h -- the analysis gates what
// Vectorize retypes, and this is what can be lowered once retyped.
//
// Declarations only: the definitions live in CombineEmit.cpp because they use
// the dragon-style macros from LegacyLLVMHelpers.h, which must not be pulled
// into other headers.
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class Location;

namespace arith {
class ConstantOp;
} // namespace arith

namespace triton {
namespace xpu {

// Must stay in sync with the TypeSwitch in emitCombineOp.
// arith::NegFOp is absent on purpose: Triton's unary minus lowers to
// `subf(0.0, x)`, so a combine region never contains one.
bool isSupportedCombineOp(Operation *op);

// A constant the region reads is usable at any stage only if one scalar value
// describes it: the same region is replayed on vector<NxT> values and on
// scalars, so the constant has to be narrowable and widenable at will.
bool isUniformConstant(Value v);

// All rejection happens here, before anything is emitted, so a region we
// cannot handle leaves no half-built IR behind. Accepts both reduce and scan
// return terminators -- the shape of the region is the same.
bool canInterpretCombine(Region &combine);

// Widen a scalar to `width` lanes *keeping its own element type*: a cmpf mask
// must become vector<Nxi1>, not vector<NxF>.
Value splatToWidth(ConversionPatternRewriter &rewriter, Location loc, Value v,
                   int64_t width);

// Always emit the constant scalar first and widen afterwards: the region may
// carry it as vector<NxT> (Vectorize rematerializes it that way) while the
// stage being lowered is scalar, or the other way round.
Value materializeConstant(ConversionPatternRewriter &rewriter, Location loc,
                          arith::ConstantOp cstOp, int64_t width);

// `width` is the lane count of the stage: > 0 while combining vector
// accumulators, 0 once the values are scalars (across-core, loop cache).
// Scalar `args` are splatted up to `width` first.
Value emitCombineOp(ConversionPatternRewriter &rewriter, Location loc,
                    Operation *op, SmallVector<Value> &args, int64_t width);

// One combine step: acc = combine(acc, cur), emitted from the region itself.
// `canInterpretCombine` must have accepted the region first. The stage width
// is inferred from the accumulators, falling back to `cur` when only they are
// vector (a scalar carry combined into vector scan chunks).
void interpretCombine(ConversionPatternRewriter &rewriter, Location loc,
                      Region &combine, SmallVector<Value> &acc, ValueRange cur);

// Fold `vecs` into their inclusive lane-prefix in place, jointly across all
// accumulators, via the masked Hillis-Steele network: each step shifts by k
// lanes (svsllp) and folds through interpretCombine. `needsMask` keeps lane<k
// at its pre-step value with the vvor masked select -- required unless the
// combine's identity is the shift's zero fill (plain add). Element type and
// lane count come from vecs[0] (all share one register-width vector shape); a
// masked combine must be f32/f16. Shared by the scan network (which then folds
// the cross-chunk carry) and the reduce collapse (which extracts the last
// lane).
void emitMaskedLanePrefix(ConversionPatternRewriter &rewriter, Location loc,
                          Region &combine, SmallVector<Value> &vecs,
                          bool needsMask);

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITONXPU_CONVERSION_COMBINE_EMIT_H
