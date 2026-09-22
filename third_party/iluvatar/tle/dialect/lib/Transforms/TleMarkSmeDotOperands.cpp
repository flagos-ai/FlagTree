#include "IR/Dialect.h"
#include "Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DEF_TRITONILUVATARTLEMARKSMEDOTOPERANDS
#include "Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

// The bit position in the launch-time SME mask is the kernel argument number of
// the base pointer, matching AccelerateMatmul's getUseSmeFlagFromPtr, so a dot
// operand marked here carries the same flag an auto-SME operand would.
static unsigned smeArgMask(Value ptr) {
  for (Value current = ptr; current;) {
    if (auto blockArg = dyn_cast<BlockArgument>(current)) {
      if (isa<tt::PointerType>(blockArg.getType()))
        return 1u << blockArg.getArgNumber();
      auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
      if (!forOp)
        return 0;
      auto init = forOp.getTiedLoopInit(blockArg);
      if (!init)
        return 0;
      current = init->get();
      continue;
    }
    Operation *def = current.getDefiningOp();
    if (!def || def->getNumOperands() == 0)
      return 0;
    // splat/broadcast/addptr/convert_layout all carry the base in operand 0.
    current = def->getOperand(0);
  }
  return 0;
}

// Views keep the encoding of the allocation they slice, so comparing roots is
// what tells whether a local_load reads a tile some SME copy wrote.
static Value getRootMemDesc(Value memDesc) {
  Value current = memDesc;
  while (Operation *def = current.getDefiningOp()) {
    if (auto subslice = dyn_cast<ttg::MemDescSubsliceOp>(def)) {
      current = subslice.getSrc();
      continue;
    }
    if (auto index = dyn_cast<ttg::MemDescIndexOp>(def)) {
      current = index.getSrc();
      continue;
    }
    if (auto trans = dyn_cast<ttg::MemDescTransOp>(def)) {
      current = trans.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(def)) {
      current = reshape.getSrc();
      continue;
    }
    break;
  }
  return current;
}

// The rowxfb8 shared layout SME writes is only GF(2)-linear up to a bit-7
// correction that LocalLoadOpConversion applies through
// isIluvatarRowXfb8LocalLoad, which requires a non-zero dot-operand useSme. No
// other element width needs the marking, so leave those layouts alone.
static bool needsRowXfb8Marking(ttg::LocalLoadOp load) {
  auto srcTy = cast<ttg::MemDescType>(load.getSrc().getType());
  if (!srcTy.getElementType().isInteger(8))
    return false;
  auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(srcTy.getEncoding());
  return shared && shared.getUseTcu() && shared.getOrder()[0] != 0;
}

static unsigned smeMaskForStagedLoad(ttg::LocalAllocOp alloc, unsigned useSme) {
  // Only the staging promote-local-store-staging created: elsewhere a zero
  // useSme can be AccelerateMatmul deliberately rejecting the load, and
  // overriding that would issue an SME transfer it already ruled out.
  if (!alloc->hasAttr("iluvatar_tle.promoted_staging"))
    return 0;
  Value src = alloc.getSrc();
  if (!src)
    return 0;
  auto load = stripConvertLayouts(src).getDefiningOp<tt::LoadOp>();
  if (!load || load.getMask() || load.getOther() || load.getIsVolatile() ||
      !load.getBoundaryCheck().empty())
    return 0;
  if (!load->hasOneUse())
    return 0;

  auto dstTy = cast<ttg::MemDescType>(alloc.getType());
  if (dstTy.getRank() != 2)
    return 0;
  auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
  if (!shared || !shared.getUseTcu())
    return 0;

  Type elemTy = dstTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return 0;
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  if (bitwidth != 8 && bitwidth != 16 && bitwidth != 32)
    return 0;

  auto ptrTy = dyn_cast<RankedTensorType>(load.getPtr().getType());
  if (!ptrTy)
    return 0;
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(ptrTy.getEncoding());
  if (!blocked || blocked.getIsSme())
    return 0;
  if (shared.getOrder()[0] != blocked.getOrder()[0])
    return 0;

  unsigned mask = smeArgMask(load.getPtr()) & useSme;
  if (!mask)
    return 0;

  auto mod = load->getParentOfType<ModuleOp>();
  unsigned contigDim = blocked.getOrder()[0];
  auto smeEnc = ttg::BlockedEncodingAttr::get(
      ptrTy.getContext(), /*isSme=*/true, /*smeMask=*/false,
      ttg::lookupNumWarps(mod), elemTy, ptrTy.getShape(), blocked.getOrder(),
      blocked.getSizePerThread(), blocked.getThreadsPerWarp(),
      blocked.getWarpsPerCTA(), ttg::TritonGPUDialect::getNumCTAs(mod));
  auto smeWpt = smeEnc.getSmeWarpsPerCTA();
  if (smeWpt.size() != 2)
    return 0;
  SmallVector<unsigned, 2> tile({16, 16});
  tile[contigDim] = 512 / bitwidth;
  for (unsigned dim = 0; dim < 2; ++dim) {
    unsigned covered = smeWpt[dim] * tile[dim];
    if (covered == 0 || ptrTy.getShape()[dim] % covered != 0)
      return 0;
  }
  return mask;
}

struct MarkSmeDotOperandsPass
    : public impl::TritonIluvatarTleMarkSmeDotOperandsBase<
          MarkSmeDotOperandsPass> {
  using impl::TritonIluvatarTleMarkSmeDotOperandsBase<
      MarkSmeDotOperandsPass>::TritonIluvatarTleMarkSmeDotOperandsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::DenseMap<Value, unsigned> smeMaskByRoot;
    module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
      if (!copy.isIluvatarSmeAsyncCopy())
        return;
      unsigned mask = smeArgMask(copy.getSrc());
      if (!mask)
        return;
      Value root = getRootMemDesc(copy.getResult());
      smeMaskByRoot[root] |= mask;
    });
    llvm::DenseMap<Value, unsigned> stagedMaskByRoot;
    module.walk([&](ttg::LocalAllocOp alloc) {
      if (unsigned mask = smeMaskForStagedLoad(alloc, useSme))
        stagedMaskByRoot[getRootMemDesc(alloc.getResult())] |= mask;
    });
    if (smeMaskByRoot.empty() && stagedMaskByRoot.empty())
      return;

    module.walk([&](ttg::LocalLoadOp load) {
      auto resultTy = dyn_cast<RankedTensorType>(load.getType());
      if (!resultTy)
        return;
      auto dot = dyn_cast<ttg::DotOperandEncodingAttr>(resultTy.getEncoding());
      if (!dot || dot.getUseSme() != 0)
        return;
      Value root = getRootMemDesc(load.getSrc());

      unsigned mask = 0;
      if (auto it = stagedMaskByRoot.find(root); it != stagedMaskByRoot.end()) {
        // The staged shape needs the flag for every element width, because
        // SmeLoad refuses to issue the SME transfer without it.
        mask = it->second;
      } else if (needsRowXfb8Marking(load)) {
        auto it = smeMaskByRoot.find(root);
        if (it != smeMaskByRoot.end())
          mask = it->second;
      }
      if (!mask)
        return;

      auto marked = ttg::DotOperandEncodingAttr::get(
          &getContext(), dot.getOpIdx(), dot.getParent(), dot.getKWidth(), mask,
          dot.getKRotate());
      load.getResult().setType(RankedTensorType::get(
          resultTy.getShape(), resultTy.getElementType(), marked));
    });
  }
};

} // namespace
} // namespace mlir::triton::iluvatar_tle
