//===- MapElementwiseExpansion.cpp - Pre-processing for map_elementwise ---===//
//
// Implements pre-processing utilities for tt.map_elementwise:
//   - Scalar-level if-conversion (scf.if → arith.select)
//   - Tensor-level expansion (lift scalar ops to tensor ops)
//   - K-way pack support via recursive binary split/join
//
// Expansion approach (tensor lifting):
//
//   All map_elementwise ops are expanded before dialect conversion. The scalar
//   region body is lifted to tensor-level ops: each scalar op becomes its
//   tensor equivalent, constants are splatted via tt.splat, and scf.if is
//   replaced with arith.select on tensors. The map_elementwise op is then
//   erased, leaving only standard tensor ops that flow through the normal
//   TritonToTileIR conversion pipeline.
//
//   For regions with control flow (scf.if), the lifting handles branch
//   flattening (scf.if → arith.select) as part of the tensor expansion.
//   For pack > 1, inputs are split into sub-element tensors via recursive
//   binary tt.split, processed independently, then reassembled via tt.join.
//
//===----------------------------------------------------------------------===//

#include "MapElementwiseExpansion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace {

using namespace mlir;

//===----------------------------------------------------------------------===//
// Tensor lifting utilities
//===----------------------------------------------------------------------===//

// Forward declaration for mutual recursion.
static LogicalResult liftBodyOp(Operation &bodyOp, OpBuilder &builder,
                                Location loc, IRMapping &mapping,
                                ArrayRef<int64_t> shape, Attribute encoding);

// Lift a scalar value to tensor: if already tensor, return as-is.
// If scalar (e.g., hoisted constant or block argument from outside),
// insert tt.splat to broadcast to the target tensor shape.
static Value liftToTensor(Value val, OpBuilder &builder, Location loc,
                          IRMapping &mapping, ArrayRef<int64_t> shape,
                          Attribute encoding) {
  Value mapped = mapping.lookupOrDefault(val);
  if (isa<RankedTensorType>(mapped.getType()))
    return mapped;
  auto tensorType = RankedTensorType::get(shape, mapped.getType(), encoding);
  auto splat = triton::SplatOp::create(builder, loc, tensorType, mapped);
  mapped = splat.getResult();
  mapping.map(val, mapped);
  return mapped;
}

// Recursively lift all ops in a region body (single block) to tensor level.
// Returns the lifted values corresponding to the scf.yield operands.
static SmallVector<Value> liftRegionBody(Region &region, OpBuilder &builder,
                                         Location loc, IRMapping &mapping,
                                         ArrayRef<int64_t> shape,
                                         Attribute encoding) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block &body = region.front();
  for (auto &op : body.without_terminator()) {
    if (failed(liftBodyOp(op, builder, loc, mapping, shape, encoding)))
      return {};
  }
  // Collect yield operands
  auto yieldOp = cast<scf::YieldOp>(body.getTerminator());
  SmallVector<Value> results;
  for (auto val : yieldOp.getOperands())
    results.push_back(
        liftToTensor(val, builder, loc, mapping, shape, encoding));
  return results;
}

// Lift a single scalar op to its tensor-level equivalent.
// Handles:
//   - arith::ConstantOp → DenseElementsAttr splat
//   - scf::IfOp → recursive expansion into arith.select
//   - generic ops → remap operands to tensor, create new op with tensor types
static LogicalResult liftBodyOp(Operation &bodyOp, OpBuilder &builder,
                                Location loc, IRMapping &mapping,
                                ArrayRef<int64_t> shape, Attribute encoding) {

  // Skip terminators handled by callers
  if (isa<triton::MapElementwiseReturnOp, scf::YieldOp>(&bodyOp))
    return success();

  // (1) arith::ConstantOp → DenseElementsAttr splat
  if (auto constOp = dyn_cast<arith::ConstantOp>(&bodyOp)) {
    auto scalarAttr = cast<TypedAttr>(constOp.getValue());
    auto tensorType =
        RankedTensorType::get(shape, scalarAttr.getType(), encoding);
    auto splatAttr = DenseElementsAttr::get(tensorType, scalarAttr);
    auto newConst = arith::ConstantOp::create(builder, loc, splatAttr);
    mapping.map(constOp.getResult(), newConst.getResult());
    return success();
  }

  // (2) scf::IfOp → lift both branches, emit arith.select
  if (auto ifOp = dyn_cast<scf::IfOp>(&bodyOp)) {
    Value condTensor = liftToTensor(ifOp.getCondition(), builder, loc, mapping,
                                    shape, encoding);

    // Process then region
    IRMapping thenMapping(mapping);
    SmallVector<Value> thenVals = liftRegionBody(
        ifOp.getThenRegion(), builder, loc, thenMapping, shape, encoding);
    if (thenVals.empty() && ifOp.getNumResults() > 0)
      return failure();

    // Process else region
    IRMapping elseMapping(mapping);
    SmallVector<Value> elseVals = liftRegionBody(
        ifOp.getElseRegion(), builder, loc, elseMapping, shape, encoding);
    if (elseVals.empty() && ifOp.getNumResults() > 0)
      return failure();

    // Emit arith.select for each result
    for (auto [i, oldRes] : llvm::enumerate(ifOp.getResults())) {
      auto sel = arith::SelectOp::create(builder, loc, condTensor, thenVals[i],
                                         elseVals[i]);
      mapping.map(oldRes, sel.getResult());
    }
    return success();
  }

  // (3) Generic op: remap operands to tensor, fix result types
  SmallVector<Value> newOperands;
  for (auto operand : bodyOp.getOperands()) {
    newOperands.push_back(
        liftToTensor(operand, builder, loc, mapping, shape, encoding));
  }

  SmallVector<Type> newResultTypes;
  for (auto result : bodyOp.getResults()) {
    newResultTypes.push_back(
        RankedTensorType::get(shape, result.getType(), encoding));
  }

  OperationState state(loc, bodyOp.getName());
  state.addOperands(newOperands);
  state.addTypes(newResultTypes);
  state.addAttributes(bodyOp.getAttrs());
  auto *newOp = Operation::create(state);
  builder.insert(newOp);

  for (auto [oldRes, newRes] :
       llvm::zip(bodyOp.getResults(), newOp->getResults())) {
    mapping.map(oldRes, newRes);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Scalar-level if-conversion (used by ifConvertMapElementwiseRegions)
//===----------------------------------------------------------------------===//

// Convert all scf.if → arith.select within a region.
// Uses walk + reverse to process innermost scf.if first, ensuring outer
// scf.if bodies are already pure when we process them.
static LogicalResult scalarIfConvert(Region &region) {
  if (!region.hasOneBlock())
    return failure();
  Block &body = region.front();

  // Collect all scf.if ops (including nested ones in then/else regions)
  SmallVector<scf::IfOp> ifOps;
  body.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  // Process innermost first (walk visits outer before inner, reverse fixes
  // this)
  for (auto ifOp : llvm::reverse(ifOps)) {
    OpBuilder builder(ifOp);
    Location loc = ifOp.getLoc();

    // Get then yield values
    auto &thenBlock = ifOp.getThenRegion().front();
    auto thenYield = cast<scf::YieldOp>(thenBlock.getTerminator());
    SmallVector<Value> thenVals(thenYield.getOperands());

    // Get else yield values
    auto &elseBlock = ifOp.getElseRegion().front();
    auto elseYield = cast<scf::YieldOp>(elseBlock.getTerminator());
    SmallVector<Value> elseVals(elseYield.getOperands());

    // Move then/else ops before the scf.if (they're pure, safe to speculate)
    for (auto &op : llvm::make_early_inc_range(thenBlock.without_terminator()))
      op.moveBefore(ifOp);
    for (auto &op : llvm::make_early_inc_range(elseBlock.without_terminator()))
      op.moveBefore(ifOp);

    // Create arith.select for each result
    SmallVector<Value> selectResults;
    for (auto [tv, ev] : llvm::zip(thenVals, elseVals)) {
      auto sel =
          arith::SelectOp::create(builder, loc, ifOp.getCondition(), tv, ev);
      selectResults.push_back(sel.getResult());
    }

    // Replace scf.if results with selects and erase
    ifOp.replaceAllUsesWith(selectResults);
    ifOp.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// K-way pack support via recursive binary split/join
//===----------------------------------------------------------------------===//

// Recursively split a tensor along its last dimension using binary tt.split.
// Input: tensor of shape [.., M, K] where K is a power of 2.
// Output: K tensors of shape [.., M].
// For K=1: just return the input (no split needed).
// For K=2: one tt.split → 2 results.
// For K>2: split into halves, recurse on each half.
static SmallVector<Value> recursiveSplit(OpBuilder &builder, Location loc,
                                         Value tensor, int K) {
  if (K == 1) {
    // Last dim is 1, drop it via reshape
    auto srcType = cast<RankedTensorType>(tensor.getType());
    auto srcShape = srcType.getShape();
    SmallVector<int64_t> newShape(srcShape.begin(), srcShape.end() - 1);
    auto newType = RankedTensorType::get(newShape, srcType.getElementType(),
                                         srcType.getEncoding());
    Value reshaped =
        triton::ReshapeOp::create(builder, loc, newType, tensor).getResult();
    return {reshaped};
  }

  if (K == 2) {
    // Base case: binary split
    auto splitOp = triton::SplitOp::create(builder, loc, tensor);
    return {splitOp.getOutLHS(), splitOp.getOutRHS()};
  }

  // Recursive case: reshape last dim K → [K/2, 2], split, recurse
  auto srcType = cast<RankedTensorType>(tensor.getType());
  auto srcShape = srcType.getShape();
  // [.., M, K] → [.., M, K/2, 2]
  SmallVector<int64_t> reshapedShape(srcShape.begin(), srcShape.end());
  reshapedShape.back() = K / 2;
  reshapedShape.push_back(2);

  auto reshapedType = RankedTensorType::get(
      reshapedShape, srcType.getElementType(), srcType.getEncoding());
  Value reshaped =
      triton::ReshapeOp::create(builder, loc, reshapedType, tensor).getResult();

  // Split [.., M, K/2, 2] → lhs: [.., M, K/2], rhs: [.., M, K/2]
  auto splitOp = triton::SplitOp::create(builder, loc, reshaped);

  // Recurse on each half
  SmallVector<Value> lhsResults =
      recursiveSplit(builder, loc, splitOp.getOutLHS(), K / 2);
  SmallVector<Value> rhsResults =
      recursiveSplit(builder, loc, splitOp.getOutRHS(), K / 2);

  // Interleave: [lhs0, rhs0, lhs1, rhs1, ...] to maintain element order
  SmallVector<Value> results;
  for (unsigned i = 0; i < lhsResults.size(); i++) {
    results.push_back(lhsResults[i]);
    results.push_back(rhsResults[i]);
  }
  return results;
}

// Recursively join K tensors into one using binary tt.join.
// Input: K tensors of shape [.., M].
// Output: tensor of shape [.., M, K].
// Reverses the recursiveSplit operation.
static Value recursiveJoin(OpBuilder &builder, Location loc,
                           ArrayRef<Value> tensors, int K,
                           RankedTensorType finalType) {
  if (K == 1) {
    // Add back last dim via reshape: [.., M] → [.., M, 1]
    auto srcType = cast<RankedTensorType>(tensors[0].getType());
    auto srcShape = srcType.getShape();
    SmallVector<int64_t> newShape(srcShape.begin(), srcShape.end());
    newShape.push_back(1);
    auto newType = RankedTensorType::get(newShape, srcType.getElementType(),
                                         srcType.getEncoding());
    return triton::ReshapeOp::create(builder, loc, newType, tensors[0])
        .getResult();
  }

  if (K == 2) {
    // Base case: binary join
    return triton::JoinOp::create(builder, loc, tensors[0], tensors[1])
        .getResult();
  }

  // Recursive case: de-interleave into even/odd halves, recurse, join
  SmallVector<Value> lhsTensors, rhsTensors;
  for (unsigned i = 0; i < tensors.size(); i += 2) {
    lhsTensors.push_back(tensors[i]);
    rhsTensors.push_back(tensors[i + 1]);
  }

  // Recurse on halves → each produces [.., M, K/2]
  Value lhs = recursiveJoin(builder, loc, lhsTensors, K / 2, finalType);
  Value rhs = recursiveJoin(builder, loc, rhsTensors, K / 2, finalType);

  // Reshape each [.., M, K/2] → [.., M * K/2] to prepare for join
  // Then join [.., M*K/2] + [.., M*K/2] → [.., M*K/2, 2]
  // Then reshape [.., M*K/2, 2] → [.., M, K]

  // Actually simpler: join adds a new last dim of 2.
  // [.., M, K/2] join → [.., M, K/2, 2]
  // Then reshape [.., M, K/2, 2] → [.., M, K]
  auto joined = triton::JoinOp::create(builder, loc, lhs, rhs);

  // Reshape to collapse the last two dims
  auto joinedType = cast<RankedTensorType>(joined.getResult().getType());
  auto joinedShape = joinedType.getShape();
  SmallVector<int64_t> collapsedShape(joinedShape.begin(),
                                      joinedShape.end() - 2);
  collapsedShape.push_back(joinedShape[joinedShape.size() - 2] * 2);

  auto collapsedType = RankedTensorType::get(
      collapsedShape, joinedType.getElementType(), joinedType.getEncoding());
  return triton::ReshapeOp::create(builder, loc, collapsedType,
                                   joined.getResult())
      .getResult();
}

//===----------------------------------------------------------------------===//
// Main expansion logic
//===----------------------------------------------------------------------===//

static bool regionHasScfIf(Region &region) {
  bool found = false;
  region.walk([&](scf::IfOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static LogicalResult expandMapElementwiseOpsImpl(Operation *rootOp) {
  SmallVector<triton::MapElementwiseOp> opsToExpand;
  rootOp->walk([&](triton::MapElementwiseOp op) { opsToExpand.push_back(op); });

  for (auto op : opsToExpand) {
    auto &region = op.getScalarOp();

    if (!region.hasOneBlock()) {
      op.emitError("map_elementwise region has multiple blocks; "
                   "expected lift-tt-cf-to-scf to have run first");
      return failure();
    }

    int pack = op.getPack();
    if (!llvm::isPowerOf2_32(pack)) {
      op.emitError("pack must be a power of 2");
      return failure();
    }

    Block &body = region.front();
    OpBuilder builder(op);
    Location loc = op.getLoc();
    unsigned numSrcs = op.getSrcs().size();

    auto inputType = cast<RankedTensorType>(op.getSrcs()[0].getType());
    auto origShape = inputType.getShape();
    auto encoding = inputType.getEncoding();

    if (pack == 1) {
      // Direct lifting on original tensor shape.
      IRMapping mapping;
      for (auto [arg, src] : llvm::zip(body.getArguments(), op.getSrcs()))
        mapping.map(arg, src);

      for (auto &bodyOp : body.without_terminator()) {
        if (failed(
                liftBodyOp(bodyOp, builder, loc, mapping, origShape, encoding)))
          return failure();
      }

      auto terminator =
          cast<triton::MapElementwiseReturnOp>(body.getTerminator());
      SmallVector<Value> results;
      for (auto val : terminator.getResult())
        results.push_back(
            liftToTensor(val, builder, loc, mapping, origShape, encoding));

      op.replaceAllUsesWith(results);
      op.erase();
    } else {
      // pack > 1: Decompose into sub-element tensors, lift, then reassemble.
      //
      // Block arg order: repeatInterleave([src0_type, src1_type, ...], pack)
      //   = [src0_e0, src0_e1, ..., src0_eK-1, src1_e0, ...]
      // Return order: repeatInterleave([out0_type, out1_type, ...], pack)
      //   = [out0_e0, out0_e1, ..., out0_eK-1, out1_e0, ...]

      int64_t lastDim = origShape.back();
      if (lastDim % pack != 0) {
        op.emitError("last dimension not divisible by pack");
        return failure();
      }

      // Step 1: Reshape each input [.., N] → [.., N/K, K]
      // Step 2: Recursively split into K sub-tensors of shape [.., N/K]
      SmallVector<SmallVector<Value>> splitResults(numSrcs);
      for (unsigned i = 0; i < numSrcs; i++) {
        Value src = op.getSrcs()[i];
        auto srcType = cast<RankedTensorType>(src.getType());

        // Build reshaped shape: [.., N/K, K]
        SmallVector<int64_t> reshapedShape(srcType.getShape().begin(),
                                           srcType.getShape().end());
        reshapedShape.back() = lastDim / pack;
        reshapedShape.push_back(pack);

        auto reshapedType = RankedTensorType::get(
            reshapedShape, srcType.getElementType(), srcType.getEncoding());
        Value reshaped =
            triton::ReshapeOp::create(builder, loc, reshapedType, src)
                .getResult();

        // Recursively split into K sub-tensors
        splitResults[i] = recursiveSplit(builder, loc, reshaped, pack);
      }

      // Compute the sub-tensor shape (used for lifting)
      SmallVector<int64_t> subShape(origShape.begin(), origShape.end());
      subShape.back() = lastDim / pack;

      // Infer sub-tensor encoding from the split result
      auto subEncoding =
          cast<RankedTensorType>(splitResults[0][0].getType()).getEncoding();

      // Step 3: Map block args to sub-tensors.
      // Block args are interleaved: [src0_e0, src0_e1, ..., src1_e0, ...]
      IRMapping mapping;
      unsigned argIdx = 0;
      for (unsigned srcIdx = 0; srcIdx < numSrcs; srcIdx++) {
        for (int p = 0; p < pack; p++) {
          mapping.map(body.getArgument(argIdx++), splitResults[srcIdx][p]);
        }
      }

      // Step 4: Lift all body ops on the sub-tensor shape [.., N/K]
      for (auto &bodyOp : body.without_terminator()) {
        if (failed(liftBodyOp(bodyOp, builder, loc, mapping, subShape,
                              subEncoding)))
          return failure();
      }

      // Step 5+6: Collect results, join K sub-tensors, reshape back.
      auto terminator =
          cast<triton::MapElementwiseReturnOp>(body.getTerminator());
      auto retVals = terminator.getResult();
      unsigned numOutputs = op.getNumResults();

      SmallVector<Value> results;
      for (unsigned outIdx = 0; outIdx < numOutputs; outIdx++) {
        // Collect the K sub-element results for this output
        SmallVector<Value> subResults;
        for (int p = 0; p < pack; p++) {
          subResults.push_back(liftToTensor(retVals[outIdx * pack + p], builder,
                                            loc, mapping, subShape,
                                            subEncoding));
        }

        // Join K sub-tensors → [.., N/K, K]
        auto outputType =
            cast<RankedTensorType>(op->getResult(outIdx).getType());
        Value joined =
            recursiveJoin(builder, loc, subResults, pack, outputType);

        // Reshape [.., N/K, K] → [.., N]
        Value result =
            triton::ReshapeOp::create(builder, loc, outputType, joined)
                .getResult();
        results.push_back(result);
      }

      op.replaceAllUsesWith(results);
      op.erase();
    }
  }

  return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult ifConvertMapElementwiseRegions(Operation *rootOp) {
  WalkResult result = rootOp->walk([&](triton::MapElementwiseOp op) {
    if (failed(scalarIfConvert(op.getScalarOp())))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult expandMapElementwiseOps(Operation *rootOp) {
  return expandMapElementwiseOpsImpl(rootOp);
}

} // namespace triton
} // namespace mlir
