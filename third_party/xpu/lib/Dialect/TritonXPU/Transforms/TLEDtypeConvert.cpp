//===----------------------------------------------------------------------===//
// TLE bf16 -> f32 promotion.
//
// XPU3 has no bf16 ALU and no bf16 <-> f32 convert instruction, so a TLE kernel
// that computes in bf16 cannot be vectorized: the arithmetic has to happen in
// f32 and the conversion has to sit on the LM boundary, where the load/store
// lowering fuses it (XPUTLETriton{Load,Store}OpConversion with
// VecBF16ToFP32{,Unordered} / VecFP32ToBF16{,Slow,Unordered}).
//
// This is the TLE counterpart of tritonxpu-dtype-convert: same promotion, but
// seeded from `tt.store` through a `tle_local_ptr` instead of
// `triton_xpu.store`, and with one extra step -- tt.load/tt.store require
// "result/value type == pointee type", so the pointer tensor is retyped to the
// promoted register type and the op is marked `xpu.lm_bf16` to record that the
// memory behind it is still bf16.
//===----------------------------------------------------------------------===//

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {
#define GEN_PASS_DEF_TRITONXPUTLEDTYPECONVERT
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// The tle_local_ptr behind a TLE load/store, when its LM buffer is bf16.
triton::xpu::TLELocalPtrOp getBf16LocalPtr(Value ptr) {
  auto lp = ptr.getDefiningOp<triton::xpu::TLELocalPtrOp>();
  if (!lp)
    return nullptr;
  auto memTy = dyn_cast<triton::gpu::MemDescType>(lp.getBuffer().getType());
  if (!memTy || !memTy.getElementType().isBF16())
    return nullptr;
  return lp;
}

bool isBf16Elem(Type t) { return getElementTypeOrSelf(t).isBF16(); }

// Only plain bf16 registers get promoted: a `!ttg.memdesc<...xbf16>` (the LM
// buffer itself) or a `!tt.ptr<bf16>` tensor also has a bf16 element type, but
// retyping those would rewrite the memory, not the computation.
bool isPromotableBf16(Type t) {
  if (!isBf16Elem(t))
    return false;
  return isa<RankedTensorType>(t) || isa<FloatType>(t);
}

// f32 counterpart of a bf16 scalar/tensor type.
Type toF32(OpBuilder &b, Type ty) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(ty))
    return RankedTensorType::get(tensorTy.getShape(), b.getF32Type(),
                                 tensorTy.getEncoding());
  return b.getF32Type();
}

// A bf16 value crossing an op with regions (a tt.reduce / tt.scan combine body)
// cannot be promoted here: the region's block arguments and body stay bf16.
// Such a tree is left alone, which just keeps the pre-pass behaviour (bf16
// compute, no vectorization).
bool hasUnpromotableRegionOp(const llvm::SetVector<Operation *> &tree) {
  return llvm::any_of(tree, [](Operation *op) {
    if (op->getNumRegions() == 0 ||
        isa<scf::ForOp, scf::IfOp, scf::YieldOp>(op))
      return false;
    return llvm::any_of(op->getResultTypes(), isPromotableBf16) ||
           llvm::any_of(op->getOperandTypes(), isPromotableBf16);
  });
}

// Promoting an op's result to f32 rewrites what its users see, so every user
// has to be inside the same tree -- or an extf to f32 (which the promotion
// makes degenerate anyway) or a store (whose pointer gets retyped below). A
// user left outside, say a bf16 tt.reduce, would end up with an f32 operand it
// cannot type.
bool hasUserOutsideTree(const llvm::SetVector<Operation *> &tree) {
  return llvm::any_of(tree, [&](Operation *op) {
    return llvm::any_of(op->getResults(), [&](Value res) {
      if (!isPromotableBf16(res.getType()))
        return false;
      return llvm::any_of(res.getUsers(), [&](Operation *user) {
        if (tree.contains(user) || isa<triton::StoreOp>(user))
          return false;
        auto ext = dyn_cast<arith::ExtFOp>(user);
        return !ext || !getElementTypeOrSelf(ext.getType()).isF32();
      });
    });
  });
}

} // namespace

struct TritonXPUTLEDtypeConvert
    : public impl::TritonXPUTLEDtypeConvertBase<TritonXPUTLEDtypeConvert> {

  using impl::TritonXPUTLEDtypeConvertBase<
      TritonXPUTLEDtypeConvert>::TritonXPUTLEDtypeConvertBase;

  // Retype the pointer tensor of a TLE load/store to the promoted register type
  // and mark the op, so the lowering still knows the memory is bf16.
  //
  // The load/store operand is not always the local_ptr DIRECTLY: the SM-gather
  // chain (encodeComputePtrs / Step1) wraps the still-unencoded tle_local_ptr
  // in a convert_layout on the way to the load, so the operand can be
  // `convert(local_ptr)`. Bailing there left the load's promoted f32 result
  // over a ptr<bf16> tensor -- "'tt.load' op failed to verify that result
  // matches ptr type" -- and killed the whole bf16 TLE pipeline. Walk the
  // convert, retype the underlying local_ptr, and rebuild the convert with the
  // promoted element type so the chain stays well-typed either way.
  void retypeBoundary(Operation *memOp, Type regElemTy) const {
    Value ptr = memOp->getOperand(0);
    auto cvt = ptr.getDefiningOp<triton::xpu::ConvertLayoutOp>();
    Value lpValue = cvt ? cvt.getSrc() : ptr;
    auto lp = getBf16LocalPtr(lpValue);
    if (!lp)
      return;
    OpBuilder b(memOp);
    auto oriTy = cast<RankedTensorType>(lp.getResult().getType());
    auto oriPtrTy = cast<triton::PointerType>(oriTy.getElementType());
    auto newTy = RankedTensorType::get(
        oriTy.getShape(),
        triton::PointerType::get(regElemTy, oriPtrTy.getAddressSpace()),
        oriTy.getEncoding());
    // $loopIndex came with the TLE stack budget work; pass the original's
    // through so a segment unroll control sliced keeps its tile index.
    auto newLp = b.create<triton::xpu::TLELocalPtrOp>(
        lp.getLoc(), newTy, lp.getBuffer(), lp.getIndices(), lp.getLoopIndex());
    Value newPtr = newLp.getResult();
    if (cvt) {
      auto cvtTy = cast<RankedTensorType>(cvt.getResult().getType());
      auto cvtPtrTy = cast<triton::PointerType>(cvtTy.getElementType());
      auto newCvtTy = RankedTensorType::get(
          cvtTy.getShape(),
          triton::PointerType::get(regElemTy, cvtPtrTy.getAddressSpace()),
          cvtTy.getEncoding());
      newPtr = b.create<triton::xpu::ConvertLayoutOp>(cvt.getLoc(), newCvtTy,
                                                      newLp.getResult());
    }
    memOp->setOperand(0, newPtr);
    memOp->setAttr("xpu.lm_bf16", b.getUnitAttr());
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Seed one op tree per bf16 TLE store.
    llvm::SetVector<Operation *> visitedOps;
    llvm::SmallVector<llvm::SetVector<Operation *>> trees;
    llvm::SmallVector<triton::StoreOp> stores;
    m.walk([&](triton::StoreOp storeOp) {
      if (visitedOps.contains(storeOp) || !getBf16LocalPtr(storeOp.getPtr()))
        return;
      llvm::SetVector<Operation *> tree;
      getOpTreeBwd(tree, visitedOps, storeOp.getValue().getDefiningOp());
      if (hasUnpromotableRegionOp(tree) || hasUserOutsideTree(tree))
        return;
      trees.emplace_back(std::move(tree));
      stores.emplace_back(storeOp);
    });
    if (stores.empty())
      return;

    // Promote the trees: bf16 constants and results become f32, bf16 values
    // that enter from outside (block arguments) get an extf, scf.yield operands
    // get a truncf so the loop-carried type is unchanged.
    for (auto &tree : trees) {
      for (Operation *op : tree) {
        OpBuilder b(op);
        auto loc = op->getLoc();

        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
          if (!isPromotableBf16(constOp.getType()))
            continue;
          SmallVector<Operation *> users(constOp.getResult().getUsers().begin(),
                                         constOp.getResult().getUsers().end());
          auto extfOp = b.create<arith::ExtFOp>(
              loc, toF32(b, constOp.getType()), constOp.getResult(),
              arith::FastMathFlagsAttr{});
          extfOp->moveAfter(constOp);
          for (Operation *user : users)
            for (auto [i, operand] : llvm::enumerate(user->getOperands()))
              if (operand == constOp.getResult())
                user->setOperand(i, extfOp.getResult());
          continue;
        }

        if (isa<scf::YieldOp>(op)) {
          for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
            if (!isPromotableBf16(operand.getType()))
              continue;
            auto truncfOp =
                b.create<arith::TruncFOp>(loc, operand.getType(), operand);
            truncfOp->moveBefore(op);
            op->setOperand(i, truncfOp.getResult());
          }
          continue;
        }

        for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
          if (operand.getDefiningOp() || !isPromotableBf16(operand.getType()))
            continue;
          auto extfOp =
              b.create<arith::ExtFOp>(loc, toF32(b, operand.getType()), operand,
                                      arith::FastMathFlagsAttr{});
          extfOp->moveBefore(op);
          op->setOperand(i, extfOp.getResult());
        }

        if (isa<scf::ForOp, scf::IfOp>(op))
          continue;
        for (Value res : op->getResults())
          if (isPromotableBf16(res.getType()))
            res.setType(toF32(b, res.getType()));
      }
    }

    // Move the conversion onto the LM boundary: the registers are f32 now, the
    // buffer is still bf16.
    // Every store that ended up with f32 registers over a bf16 buffer, not just
    // the seeded ones: two stores can share a promoted subexpression.
    m.walk([&](triton::StoreOp storeOp) {
      Type regElemTy = getElementTypeOrSelf(storeOp.getValue().getType());
      if (regElemTy.isF32())
        retypeBoundary(storeOp, regElemTy);
    });
    m.walk([&](triton::LoadOp loadOp) {
      Type regElemTy = getElementTypeOrSelf(loadOp.getResult().getType());
      if (regElemTy.isF32())
        retypeBoundary(loadOp, regElemTy);
    });

    // The extf/truncf that used to do the conversion are degenerate now.
    m.walk([&](arith::ExtFOp extfOp) {
      if (getElementTypeOrSelf(extfOp.getIn().getType()) ==
          getElementTypeOrSelf(extfOp.getType())) {
        extfOp.getOut().replaceAllUsesWith(extfOp.getIn());
        extfOp.erase();
      }
    });
    m.walk([&](arith::TruncFOp truncfOp) {
      if (getElementTypeOrSelf(truncfOp.getIn().getType()) ==
          getElementTypeOrSelf(truncfOp.getType())) {
        truncfOp.getOut().replaceAllUsesWith(truncfOp.getIn());
        truncfOp.erase();
      }
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
