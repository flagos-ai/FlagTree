#include "TritonILUVATARGPUTransforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONILUVATARGPUCHAINDOTKROTATE
#include "TritonILUVATARGPUTransforms/Passes.h.inc"

namespace ttg = triton::gpu;

namespace {

// In a chain dot the B operand is a previous dot's accumulator, so it arrives
// in the #mma tile while the TCU wants the #dot_op tile. The two differ only in
// where the low three K bits sit, and K is the reduced axis, so relabeling it
// is exact as long as BOTH operands are relabeled the same way. That is what
// the kRotate flag asks iluvatarDotTile to do, and rotated, the opIdx=1 tile
// *is* the accumulator tile. So the convert collapses to an identity instead of
// the ~17 warp shuffles the generic transferWithinWarp needs -- on every
// iteration of a flash-attention KV loop. This is the LinearLayout-era
// replacement for v3.2's isMmaToDotShortcutForB + kSwizzle pair.
//
// A pays for the rotation only in the order it reads K out of shared memory,
// and LocalLoadOpConversion derives those addresses from the layout, so no
// lowering code changes.

// Only the fp16/bf16 tile (kWidth == 2) has a derived rotation.
ttg::DotOperandEncodingAttr getRotatableEncoding(Value operand,
                                                 unsigned expectedOpIdx) {
  auto ty = dyn_cast<RankedTensorType>(operand.getType());
  if (!ty)
    return {};
  auto enc = dyn_cast<ttg::DotOperandEncodingAttr>(ty.getEncoding());
  if (!enc || enc.getOpIdx() != expectedOpIdx || enc.getKWidth() != 2 ||
      enc.getKRotate() != 0 ||
      !isa<ttg::IluvatarMmaEncodingAttr>(enc.getParent()))
    return {};
  return enc;
}

ttg::DotOperandEncodingAttr rotate(ttg::DotOperandEncodingAttr enc) {
  return ttg::DotOperandEncodingAttr::get(enc.getContext(), enc.getOpIdx(),
                                          enc.getParent(), enc.getKWidth(),
                                          enc.getUseSme(), /*kRotate=*/1);
}

// Retyping the defining op in place is what makes this cheap: for B that is the
// convert we are trying to erase, for A a local_load or convert whose lowering
// reads the layout. Any other producer (a block argument, a dot, ...) is left
// alone rather than fixed up.
bool canRetypeInPlace(Value operand) {
  Operation *def = operand.getDefiningOp();
  return def && def->hasOneUse() &&
         isa<ttg::ConvertLayoutOp, ttg::LocalLoadOp>(def);
}

bool rewriteDot(triton::DotOp dot) {
  auto aEnc = getRotatableEncoding(dot.getA(), /*opIdx=*/0);
  auto bEnc = getRotatableEncoding(dot.getB(), /*opIdx=*/1);
  if (!aEnc || !bEnc || aEnc.getParent() != bEnc.getParent())
    return false;
  if (!canRetypeInPlace(dot.getA()) || !canRetypeInPlace(dot.getB()))
    return false;

  // Only worth it if rotating actually turns B's convert into an identity,
  // which additionally requires the producing #mma to distribute warps the same
  // way B does (all warps on N -- what TCUWarpsPerTile's warpsBiasN sets up for
  // a detected chain). Comparing the layouts outright covers that and every
  // other shape/warp precondition at once.
  auto bCvt = dot.getB().getDefiningOp<ttg::ConvertLayoutOp>();
  if (!bCvt)
    return false;
  auto srcTy = cast<RankedTensorType>(bCvt.getSrc().getType());
  if (!isa<ttg::IluvatarMmaEncodingAttr>(srcTy.getEncoding()))
    return false;
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  if (ttg::toLinearLayout(srcTy) !=
      ttg::toLinearLayout(bTy.getShape(), rotate(bEnc)))
    return false;

  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  dot.getA().setType(aTy.cloneWithEncoding(rotate(aEnc)));
  dot.getB().setType(bTy.cloneWithEncoding(rotate(bEnc)));
  return true;
}

} // anonymous namespace

class TritonILUVATARGPUChainDotKRotatePass
    : public impl::TritonILUVATARGPUChainDotKRotateBase<
          TritonILUVATARGPUChainDotKRotatePass> {

public:
  void runOnOperation() override {
    getOperation().walk([](triton::DotOp dot) { rewriteDot(dot); });
  }
};

std::unique_ptr<Pass> createTritonILUVATARGPUChainDotKRotatePass() {
  return std::make_unique<TritonILUVATARGPUChainDotKRotatePass>();
}

} // namespace mlir
