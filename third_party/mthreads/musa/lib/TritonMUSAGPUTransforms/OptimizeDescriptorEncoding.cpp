#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MemDescUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <unordered_set>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

inline constexpr llvm::StringLiteral kHostTensorDescABIArgsAttr =
    "musa.host_tensordesc_abi_args";

struct UseInfo {
  TypedValue<tt::TensorDescType> descriptor;
  Operation *use;
  Attribute desiredSharedEncoding;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> allocShape;
  ttg::CGAEncodingAttr cgaLayout;
};

static bool isTMACompatibleEncoding(Attribute enc) {
  if (isa_and_nonnull<ttg::SwizzledSharedEncodingAttr>(enc))
    return true;
  if (auto nvmma = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(enc))
    return !nvmma.getTransposed() && !nvmma.getFp4Padded();
  return false;
}

static Attribute findDirectLoadEncodingFromUsers(Operation *op) {
  for (Operation *user : op->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      auto enc = alloc.getType().getEncoding();
      if (isTMACompatibleEncoding(enc))
        return enc;
    } else if (auto store = dyn_cast<ttg::LocalStoreOp>(user)) {
      auto dstTy = dyn_cast<ttg::MemDescType>(store.getDst().getType());
      auto enc = dstTy ? dstTy.getEncoding() : Attribute();
      if (isTMACompatibleEncoding(enc))
        return enc;
    }
  }
  return {};
}

static Attribute canonicalizeDesiredSharedEncoding(
    Operation *op, TypedValue<tt::TensorDescType> desc, Attribute encoding,
    ArrayRef<int64_t> shape, unsigned numCTAs) {
  if (!encoding)
    return {};
  auto blockTy = desc.getType().getSignlessBlockType();
  auto usageShape = shape.empty()
                        ? SmallVector<int64_t>(blockTy.getShape().begin(),
                                               blockTy.getShape().end())
                        : SmallVector<int64_t>(shape.begin(), shape.end());
  auto usageTy = RankedTensorType::get(usageShape, blockTy.getElementType());
  auto canonical =
      triton::musa::tryMapTMECompatibleSharedEncodingToCanonicalSwizzled(
          op, usageTy, encoding, usageShape, numCTAs);
  return canonical ? Attribute(*canonical) : Attribute();
}

static Attribute findDescriptorLoadEncoding(tt::DescriptorLoadOp loadOp) {
  auto landingTy =
      triton::musa::getUniqueCanonicalLandingRootMemDescType(loadOp);
  if (!landingTy)
    return {};
  Attribute enc = landingTy->getEncoding();
  if (isTMACompatibleEncoding(enc))
    return enc;
  return {};
}

static SmallVector<int64_t>
findDescriptorLoadAllocShape(tt::DescriptorLoadOp loadOp) {
  auto landingTy =
      triton::musa::getUniqueCanonicalLandingRootMemDescType(loadOp);
  if (!landingTy)
    return {};
  auto allocShape = landingTy->getAllocShape();
  if (allocShape.empty())
    return {};
  auto rank = loadOp.getDesc().getType().getBlockType().getRank();
  SmallVector<int64_t> result(rank, 1);
  assert(allocShape.size() <= static_cast<size_t>(rank));
  auto rankDiff = rank - static_cast<int>(allocShape.size());
  std::copy(allocShape.begin(), allocShape.end(), result.begin() + rankDiff);
  return result;
}

static SmallVector<int64_t> expandToRank(ArrayRef<int64_t> shape, int rank) {
  SmallVector<int64_t> result(rank, 1);
  assert(shape.size() <= static_cast<size_t>(rank));
  auto rankDiff = rank - static_cast<int>(shape.size());
  std::copy(shape.begin(), shape.end(), result.begin() + rankDiff);
  return result;
}

static std::optional<UseInfo> getUseInfo(Operation *op, unsigned numCTAs) {
  UseInfo info;
  info.use = op;
  if (auto load = dyn_cast<tt::DescriptorLoadOp>(op)) {
    info.descriptor = load.getDesc();
    auto shape = load.getResult().getType().getShape();
    auto rank = load.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    auto rawEncoding = findDescriptorLoadEncoding(load);
    if (!rawEncoding)
      rawEncoding = load.getDesc().getType().getBlockType().getEncoding();
    info.desiredSharedEncoding = canonicalizeDesiredSharedEncoding(
        load, info.descriptor, rawEncoding, info.shape, numCTAs);
    info.allocShape = findDescriptorLoadAllocShape(load);
    auto encoding =
        info.desiredSharedEncoding ? info.desiredSharedEncoding : rawEncoding;
    info.cgaLayout =
        encoding ? ttg::getCGALayout(encoding) : ttg::CGAEncodingAttr();
    return info;
  }
  if (auto gather = dyn_cast<tt::DescriptorGatherOp>(op)) {
    info.descriptor = gather.getDesc();
    auto shape = gather.getResult().getType().getShape();
    auto rank = gather.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    auto rawEncoding = findDirectLoadEncodingFromUsers(op);
    if (!rawEncoding)
      rawEncoding = gather.getDesc().getType().getBlockType().getEncoding();
    info.desiredSharedEncoding = canonicalizeDesiredSharedEncoding(
        gather, info.descriptor, rawEncoding, info.shape, numCTAs);
    auto encoding =
        info.desiredSharedEncoding ? info.desiredSharedEncoding : rawEncoding;
    info.cgaLayout =
        encoding ? ttg::getCGALayout(encoding) : ttg::CGAEncodingAttr();
    return info;
  }
  if (auto store = dyn_cast<tt::DescriptorStoreLikeOpInterface>(op)) {
    info.descriptor = store.getDesc();
    auto shape = store.getSrc().getType().getShape();
    auto rank = store.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    auto rawEncoding = store.getDesc().getType().getBlockType().getEncoding();
    info.desiredSharedEncoding =
        canonicalizeDesiredSharedEncoding(store.getOperation(), info.descriptor,
                                          rawEncoding, info.shape, numCTAs);
    auto encoding =
        info.desiredSharedEncoding ? info.desiredSharedEncoding : rawEncoding;
    info.cgaLayout =
        encoding ? ttg::getCGALayout(encoding) : ttg::CGAEncodingAttr();
    return info;
  }
  return std::nullopt;
}

struct EncodingInfo {
  Attribute desiredEncoding;
  ttg::CGAEncodingAttr cgaLayout;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> allocShape;
  bool forcedToDefault = false;

  bool operator==(const EncodingInfo &other) const {
    return desiredEncoding == other.desiredEncoding &&
           cgaLayout == other.cgaLayout &&
           forcedToDefault == other.forcedToDefault && shape == other.shape &&
           allocShape == other.allocShape;
  }
};

} // namespace

template <> struct std::hash<EncodingInfo> {
  size_t operator()(const EncodingInfo &einfo) const {
    return llvm::hash_combine(einfo.desiredEncoding, einfo.cgaLayout,
                              einfo.forcedToDefault,
                              llvm::ArrayRef<int64_t>(einfo.shape),
                              llvm::ArrayRef<int64_t>(einfo.allocShape));
  }
};

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUOPTIMIZEDESCRIPTORENCODING
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

static const EncodingInfo *internEncoding(std::unordered_set<EncodingInfo> &set,
                                          EncodingInfo info) {
  return &*set.insert(std::move(info)).first;
}

static EncodingInfo combineEncodings(const EncodingInfo &lhs,
                                     const EncodingInfo &rhs, unsigned rank) {
  EncodingInfo result;
  result.forcedToDefault = lhs.forcedToDefault || rhs.forcedToDefault;
  if (result.forcedToDefault)
    return result;

  if (lhs.shape.empty() || lhs.shape == rhs.shape)
    result.shape = rhs.shape;
  else if (rhs.shape.empty())
    result.shape = lhs.shape;
  else {
    assert(lhs.shape.size() == rhs.shape.size());
    result.shape.reserve(lhs.shape.size());
    for (auto [lhsDim, rhsDim] : llvm::zip_equal(lhs.shape, rhs.shape))
      result.shape.push_back(std::min(lhsDim, rhsDim));
  }

  if (lhs.allocShape.empty() || lhs.allocShape == rhs.allocShape)
    result.allocShape = rhs.allocShape;
  else if (rhs.allocShape.empty())
    result.allocShape = lhs.allocShape;
  else
    result.forcedToDefault = true;

  llvm::SetVector<ttg::CGAEncodingAttr> cgaLayouts;
  if (lhs.cgaLayout)
    cgaLayouts.insert(lhs.cgaLayout);
  if (rhs.cgaLayout)
    cgaLayouts.insert(rhs.cgaLayout);

  auto getDefaultLayout = [&](ttg::CGAEncodingAttr encoding) {
    auto *ctx = encoding.getContext();
    auto kBlock = StringAttr::get(ctx, "block");
    auto dims = triton::standardOutDimNames(ctx, rank);
    auto numCTAs = encoding.getLinearLayout().getInDimSize(kBlock);
    triton::LinearLayout llDefault;
    for (unsigned i = 0; i < rank - 1; ++i)
      llDefault *= triton::LinearLayout::identity1D(1, kBlock, dims[i]);
    llDefault *= triton::LinearLayout::identity1D(numCTAs, kBlock, dims.back());
    return ttg::CGAEncodingAttr::get(ctx, llDefault);
  };

  switch (cgaLayouts.size()) {
  case 2:
    result.cgaLayout = getDefaultLayout(lhs.cgaLayout);
    break;
  case 1:
    result.cgaLayout = cgaLayouts[0];
    break;
  default:
    break;
  }

  llvm::SetVector<Attribute> desiredEncodings;
  if (lhs.desiredEncoding)
    desiredEncodings.insert(lhs.desiredEncoding);
  if (rhs.desiredEncoding)
    desiredEncodings.insert(rhs.desiredEncoding);

  switch (desiredEncodings.size()) {
  case 2:
    result.forcedToDefault = true;
    break;
  case 1:
    result.desiredEncoding = desiredEncodings[0];
    break;
  default:
    break;
  }
  return result;
}

static tt::TensorDescType
getTensorDescTypeWithEncoding(Operation *op, RankedTensorType blockTy,
                              Attribute encoding) {
  auto sharedEnc = cast<ttg::SharedEncodingTrait>(encoding);
  auto updatedEncoding = ttng::updateEncodingForShape(op, sharedEnc, blockTy);
  return tt::TensorDescType::get(blockTy.getContext(),
                                 blockTy.cloneWithEncoding(updatedEncoding));
}

static void updateFunctionType(tt::FuncOp func) {
  SmallVector<Type> argTys(func.getBody().front().getArgumentTypes());
  SmallVector<Type> resultTys(func.getResultTypes());
  func.setFunctionType(FunctionType::get(func.getContext(), argTys, resultTys));
}

static void assignMemoryLayouts(tt::FuncOp func) {
  std::unordered_set<EncodingInfo> encodings;
  llvm::MapVector<TypedValue<tt::TensorDescType>, const EncodingInfo *>
      valueToEncodingInfo;
  llvm::PriorityWorklist<TypedValue<tt::TensorDescType>> worklist;
  unsigned numCTAs =
      std::max(1u, static_cast<unsigned>(ttg::lookupNumCTAs(func)));

  auto updateEncoding = [&](ArrayRef<Value> descValues, EncodingInfo info) {
    for (Value value : descValues) {
      auto typedVal = cast<TypedValue<tt::TensorDescType>>(value);
      auto it = valueToEncodingInfo.find(typedVal);
      if (it != valueToEncodingInfo.end()) {
        info = combineEncodings(*it->second, info,
                                typedVal.getType().getBlockType().getRank());
      }
    }

    auto *einfo = internEncoding(encodings, std::move(info));
    for (Value value : descValues) {
      auto typedVal = cast<TypedValue<tt::TensorDescType>>(value);
      auto res = valueToEncodingInfo.try_emplace(typedVal, einfo);
      if (res.second) {
        worklist.insert(typedVal);
      } else if (res.first->second != einfo) {
        res.first->second = einfo;
        worklist.insert(typedVal);
      }
    }
  };

  bool isKernel = triton::isKernel(func);
  for (auto blockArg : func.getBlocks().front().getArguments()) {
    if (auto desc = dyn_cast<TypedValue<tt::TensorDescType>>(blockArg)) {
      updateEncoding(
          {desc}, EncodingInfo{{}, {}, {}, {}, /*forcedToDefault=*/!isKernel});
    }
  }

  func.walk([&](Operation *op) {
    if (auto info = getUseInfo(op, numCTAs)) {
      updateEncoding(info->descriptor,
                     EncodingInfo{info->desiredSharedEncoding, info->cgaLayout,
                                  info->shape, info->allocShape});
      return;
    }

    bool forcedToDefault =
        isa<tt::CallOp, tt::ReturnOp, ttng::ReinterpretTensorDescOp>(op);
    auto *einfo = internEncoding(encodings,
                                 EncodingInfo{{}, {}, {}, {}, forcedToDefault});

    auto seedEncoding = [&](Value value) {
      auto typedVal = cast<TypedValue<tt::TensorDescType>>(value);
      valueToEncodingInfo.try_emplace(typedVal, einfo);
      if (forcedToDefault)
        worklist.insert(typedVal);
    };

    for (Value result : op->getResults()) {
      if (auto desc = dyn_cast<TypedValue<tt::TensorDescType>>(result))
        seedEncoding(desc);
    }
    for (Value operand : op->getOperands()) {
      if (auto desc = dyn_cast<TypedValue<tt::TensorDescType>>(operand))
        seedEncoding(desc);
    }
  });

  while (!worklist.empty()) {
    auto desc = worklist.pop_back_val();

    for (OpOperand &use : desc.getUses()) {
      Operation *op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        updateEncoding(triton::getTiedArgs(op, use.getOperandNumber() - offset),
                       EncodingInfo{});
      } else if (isa<scf::YieldOp>(op)) {
        updateEncoding(
            triton::getTiedArgs(op->getParentOp(), use.getOperandNumber()),
            EncodingInfo{});
      }
    }

    if (auto opResult = dyn_cast<OpResult>(desc)) {
      Operation *definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp))
        updateEncoding(
            triton::getTiedArgs(definingOp, opResult.getResultNumber()),
            EncodingInfo{});
    } else if (auto blockArg = dyn_cast<BlockArgument>(desc)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        updateEncoding(
            triton::getTiedArgs(parentOp, blockArg.getArgNumber() - offset),
            EncodingInfo{});
      }
    }
  }

  for (auto &[desc, einfo] : valueToEncodingInfo) {
    auto existingTy = desc.getType().getBlockType();
    auto preferredCGA = einfo->cgaLayout;
    if (!preferredCGA && existingTy.getEncoding())
      preferredCGA = ttg::getCGALayout(existingTy.getEncoding());
    auto newEncoding =
        triton::musa::normalizeTMECompatibleSharedEncodingOrDefault(
            desc.getDefiningOp(), existingTy, einfo->desiredEncoding,
            preferredCGA, einfo->shape, einfo->allocShape, numCTAs);
    desc.setType(getTensorDescTypeWithEncoding(desc.getDefiningOp(), existingTy,
                                               newEncoding));
  }

  SmallVector<Type> argTys(func.getBody().front().getArgumentTypes());
  SmallVector<Type> resultTys(func.getResultTypes());
  for (auto [i, resultTy] : llvm::enumerate(resultTys)) {
    auto descTy = dyn_cast<tt::TensorDescType>(resultTy);
    if (!descTy)
      continue;
    auto encoding = triton::musa::getDefaultTMECompatibleSharedEncoding(
        descTy.getBlockType(), {}, {}, numCTAs);
    resultTys[i] =
        getTensorDescTypeWithEncoding(nullptr, descTy.getBlockType(), encoding);
  }
  func.setFunctionType(FunctionType::get(func.getContext(), argTys, resultTys));
}

static bool matchesExpandedTensorDescABI(Block &entry, unsigned descArgIdx,
                                         unsigned rank) {
  unsigned suffixCount = rank * 2;
  if (descArgIdx + suffixCount >= entry.getNumArguments())
    return false;
  for (unsigned i = 0; i < rank; ++i) {
    Type argTy = entry.getArgument(descArgIdx + 1 + i).getType();
    if (!argTy.isSignlessInteger(32))
      return false;
  }
  for (unsigned i = 0; i < rank; ++i) {
    Type argTy = entry.getArgument(descArgIdx + 1 + rank + i).getType();
    if (!argTy.isSignlessInteger(64))
      return false;
  }
  return true;
}

static void compactUnusedHostTensorDescABI(tt::FuncOp func) {
  if (!triton::isKernel(func))
    return;

  Block &entry = func.getBody().front();
  SmallVector<unsigned> descABIArgs;
  llvm::BitVector eraseMask(entry.getNumArguments());
  auto emptyAttr = DictionaryAttr::get(func.getContext());
  SmallVector<DictionaryAttr> allArgAttrs;
  allArgAttrs.reserve(entry.getNumArguments());
  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    DictionaryAttr attr = func.getArgAttrDict(i);
    allArgAttrs.push_back(attr ? attr : emptyAttr);
  }

  for (unsigned i = 0; i < entry.getNumArguments();) {
    auto descTy = dyn_cast<tt::TensorDescType>(entry.getArgument(i).getType());
    if (!descTy) {
      ++i;
      continue;
    }

    unsigned rank = std::max<int64_t>(1, descTy.getBlockType().getRank());
    unsigned abiArgs = 1;
    if (matchesExpandedTensorDescABI(entry, i, rank)) {
      abiArgs = 1 + 2 * rank;
      bool suffixUnused = true;
      for (unsigned j = 0; j < 2 * rank; ++j) {
        if (!entry.getArgument(i + 1 + j).use_empty()) {
          suffixUnused = false;
          break;
        }
      }
      if (suffixUnused) {
        abiArgs = 1;
        for (unsigned j = 0; j < 2 * rank; ++j)
          eraseMask.set(i + 1 + j);
      }
    }
    descABIArgs.push_back(abiArgs);
    i += abiArgs == 1 ? 1 : abiArgs;
  }

  if (eraseMask.any()) {
    entry.eraseArguments(eraseMask);
    updateFunctionType(func);
    if (!allArgAttrs.empty()) {
      SmallVector<DictionaryAttr> newArgAttrs;
      newArgAttrs.reserve(entry.getNumArguments());
      for (auto [i, attr] : llvm::enumerate(allArgAttrs)) {
        if (!eraseMask.test(i))
          newArgAttrs.push_back(attr);
      }
      func.setAllArgAttrs(newArgAttrs);
    }
  }

  unsigned descIdx = 0;
  for (auto [argIdx, arg] : llvm::enumerate(entry.getArguments())) {
    if (!isa<tt::TensorDescType>(arg.getType()))
      continue;
    func.setArgAttr(argIdx, kHostTensorDescABIArgsAttr,
                    IntegerAttr::get(IntegerType::get(func.getContext(), 32),
                                     descABIArgs[descIdx++]));
  }
}

// Rewrite:
//   convert_layout(fp_to_fp(x)) #dot_operand
// ->
//   fp_to_fp(convert_layout(x) #dot_operand)
//
// For descriptor-driven fp8/f16 dot paths this keeps conversion adjacent to
// descriptor loads and avoids a later blocked->dot shmem staging round-trip.
class HoistFpToFpAcrossDotOperandConvert
    : public OpRewritePattern<ttg::ConvertLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::ConvertLayoutOp cvt,
                                PatternRewriter &rewriter) const override {
    auto dstTy = dyn_cast<RankedTensorType>(cvt.getType());
    if (!dstTy)
      return failure();
    if (!isa_and_nonnull<ttg::DotOperandEncodingAttr>(dstTy.getEncoding()))
      return failure();

    auto fpToFp = cvt.getSrc().getDefiningOp<tt::FpToFpOp>();
    if (!fpToFp)
      return failure();

    auto midTy = dyn_cast<RankedTensorType>(fpToFp.getType());
    auto srcTy = dyn_cast<RankedTensorType>(fpToFp.getSrc().getType());
    if (!midTy || !srcTy)
      return failure();
    if (midTy.getShape() != dstTy.getShape() ||
        srcTy.getShape() != dstTy.getShape())
      return failure();
    if (midTy.getElementType() != dstTy.getElementType() ||
        srcTy.getElementType() == dstTy.getElementType())
      return failure();

    auto newCvtTy = RankedTensorType::get(
        dstTy.getShape(), srcTy.getElementType(), dstTy.getEncoding());
    rewriter.setInsertionPoint(cvt);
    Value newCvt = ttg::ConvertLayoutOp::create(rewriter, cvt.getLoc(),
                                                newCvtTy, fpToFp.getSrc());
    Value newFpToFp =
        tt::FpToFpOp::create(rewriter, fpToFp.getLoc(), dstTy, newCvt);
    rewriter.replaceOp(cvt, newFpToFp);
    return success();
  }
};

} // namespace

struct TritonMUSAGPUOptimizeDescriptorEncodingPass
    : impl::TritonMUSAGPUOptimizeDescriptorEncodingBase<
          TritonMUSAGPUOptimizeDescriptorEncodingPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (Operation &op : mod.getBodyRegion().front()) {
      auto func = dyn_cast<tt::FuncOp>(&op);
      if (!func)
        continue;
      assignMemoryLayouts(func);
      compactUnusedHostTensorDescABI(func);
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<HoistFpToFpAcrossDotOperandConvert>(&getContext());
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir
