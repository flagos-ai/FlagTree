//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
// TLE → Linalg/Tensor lowering patterns:
//   tle.dsl_region → spine_ext.raw_region
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TLEToLinalg/TLEToLinalg.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tle-to-linalg"

using namespace mlir;
// Alias onto the generic third_party/tle dialect namespace so the
// pattern code reads unchanged.
namespace tle = ::mlir::triton::tle;

namespace {

// ============================================================================
// DSLRegionOpPattern: tle.dsl_region → spine_ext.raw_region
//
// Reads the op's real region body (raw fn ops, built at trace time by
// create_tle_dsl_region_direct — no string attr to parse), clones it into a
// new generic (unregistered) "spine_ext.raw_region" op that spine-opt
// processes via SpineRawRegionInlinePass, replacing func.return →
// spine_ext.return.
// ============================================================================
struct DSLRegionOpPattern : public OpRewritePattern<tle::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tle::DSLRegionOp op,
                                PatternRewriter &rewriter) const override {
    // 0. Positional-anchor placeholder. call_registry.py emits an empty
    //    tle.dsl_region named "__spine_bridge_pt_N" at the exact program point
    //    of each llvm-direct _sr_call, so svector and bridge stages interleave
    //    in any order. Lower it to a func.call to a private no-arg stub; the
    //    stub survives to ll.mlir as `llvm.call @__spine_bridge_pt_N`, which
    //    _inject_mixed_llvm_llmlir then text-replaces with the real bridge.
    //    fn_name is a free extra attribute on the generic DSLRegionOp (its
    //    ODS has no fn_name operand), set at emit time in
    //    create_tle_dsl_region_direct.
    auto fnNameAttr = op->getAttrOfType<StringAttr>("fn_name");
    if (!fnNameAttr)
      return op.emitError("tle.dsl_region: missing fn_name attribute");
    StringRef fnName = fnNameAttr.getValue();
    if (fnName.size() >= 18 && fnName.substr(0, 18) == "__spine_bridge_pt_") {
      auto loc = op.getLoc();
      auto mod = op->getParentOfType<ModuleOp>();
      if (mod && !mod.lookupSymbol(fnName)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto fnType = FunctionType::get(rewriter.getContext(), {}, {});
        auto decl = func::FuncOp::create(rewriter, loc, fnName, fnType);
        decl.setSymVisibilityAttr(
            StringAttr::get(rewriter.getContext(), "private"));
        Block *body = decl.addEntryBlock();
        OpBuilder declBuilder(body, body->end());
        func::ReturnOp::create(declBuilder, loc);
      }
      func::CallOp::create(rewriter, loc, fnName, TypeRange{}, ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    // 1. The raw fn body is already a real region on the op; its block args are
    //    the raw fn parameters (memref<*> etc.).
    Region &srcRegion = op.getBody();
    if (srcRegion.empty())
      return op.emitError("tle.dsl_region: empty region");
    Block &srcBlock = srcRegion.front();
    auto argTypes = srcBlock.getArgumentTypes();

    // 2. Build spine_ext.raw_region as a generic (unregistered) op.
    //    The ptr->memref pipeline wraps tle.dsl_region's !tt.ptr operands in a
    //    cast chain (ptr.to_ptr <- memref.reinterpret_cast <- %arg :
    //    memref<*>), because dsl_region is not part of those passes' conversion
    //    target. Trace each operand back through that chain to the value whose
    //    type matches the raw fn's block-arg type (the original memref<*>), so
    //    the raw_region operand types line up with the region block args.
    OperationState state(op.getLoc(), "spine_ext.raw_region");
    SmallVector<Value> operands;
    unsigned idx = 0;
    for (Value in : op.getInputs()) {
      Type want = idx < argTypes.size() ? argTypes[idx] : Type();
      // Walk def chain through the cast ops the ptr pipeline inserts.
      for (int hop = 0; hop < 8 && in.getType() != want; ++hop) {
        Operation *def = in.getDefiningOp();
        if (!def)
          break;
        if (auto c = dyn_cast<UnrealizedConversionCastOp>(def)) {
          if (c.getInputs().size() != 1)
            break;
          in = c.getInputs().front();
        } else if (def->getName().getStringRef() == "ptr.to_ptr" &&
                   def->getNumOperands() == 1) {
          in = def->getOperand(0);
        } else if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
          in = rc.getSource();
        } else if (auto mc = dyn_cast<memref::CastOp>(def)) {
          in = mc.getSource();
        } else {
          break;
        }
      }
      operands.push_back(in);
      ++idx;
    }
    state.addOperands(operands);
    state.addAttribute("fn_name", fnNameAttr);

    // 3. Build region with an empty block first; create the op so the region
    //    is attached to a container BEFORE cloning into it (cloning calls
    //    Region::getContext(), which asserts on a detached region).
    Region *body = state.addRegion();
    Block *block = new Block();
    body->push_back(block);
    for (Type paramTy : argTypes)
      block->addArgument(paramTy, op.getLoc());

    Operation *newOp = rewriter.create(state);

    // 4. Clone the raw fn body into the now-attached region block, replacing
    //    func.return → spine_ext.return.
    Block *attached = &newOp->getRegion(0).front();
    IRMapping mapping;
    for (auto [fArg, bArg] :
         llvm::zip(srcBlock.getArguments(), attached->getArguments()))
      mapping.map(fArg, bArg);

    OpBuilder bodyBuilder(attached, attached->end());
    for (Operation &inner : srcBlock) {
      if (isa<func::ReturnOp>(inner)) {
        OperationState retState(inner.getLoc(), "spine_ext.return");
        bodyBuilder.create(retState);
      } else {
        bodyBuilder.clone(inner, mapping);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

// ============================================================================
// Pattern registration
// ============================================================================
void mlir::triton::populateTLEToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<DSLRegionOpPattern>(patterns.getContext());
}
