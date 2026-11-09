#ifndef TRITON_HCU_MATRIX_DESCRIPTOR_LAYOUT_H
#define TRITON_HCU_MATRIX_DESCRIPTOR_LAYOUT_H

#include "Utility.h"
#include "../TritonHCUGPUToLLVM/FiniteSharedReadPlan.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir::triton::HCU {

inline bool isAsyncMatrixDescriptor(triton::MatrixLoadOp op) {
  auto async = op->getAttrOfType<BoolAttr>("tt.load.async");
  return async && async.getValue();
}

inline bool isSyncMatrixDescriptor(triton::MatrixLoadOp op) {
  auto async = op->getAttrOfType<BoolAttr>("tt.load.async");
  return async && !async.getValue();
}

// Use the same complete address relation and instruction grouping as native
// lowering. The vector-only bank model does not describe the implicit gather
// of gfx936 matrix reads. This remains a structural cost, not a latency claim.
inline std::optional<std::pair<uint64_t, unsigned>>
getDescriptorReadBankCost(RankedTensorType readType,
                          gpu::SharedLinearEncodingAttr shared) {
  auto *ctx = readType.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto block = StringAttr::get(ctx, "block");
  auto offset = StringAttr::get(ctx, "offset");
  auto cvt = gpu::toLinearLayout(readType).invertAndCompose(
      gpu::toLinearLayout(readType.getShape(), shared));
  if (!cvt.isTrivialOver({block}))
    return std::nullopt;
  cvt = cvt.sublayout({reg, lane, warp}, {offset});
  unsigned registers = cvt.getInDimSize(reg), waves = cvt.getInDimSize(warp);
  if (cvt.getInDimSize(lane) != 64 ||
      uint64_t(registers) * waves * 64 > (uint64_t(1) << 20))
    return std::nullopt;
  SmallVector<unsigned> physical;
  for (unsigned r = 0; r < registers; ++r)
    for (unsigned w = 0; w < waves; ++w)
      for (unsigned l = 0; l < 64; ++l)
        physical.push_back(cvt.apply({{reg, int32_t(r)}, {lane, int32_t(l)},
                                     {warp, int32_t(w)}})[0].second);
  auto plan = planFiniteSharedReads(registers, waves, cvt.getOutDimSize(offset),
                                   physical, true, true, true);
  if (!plan || !plan->matrix || plan->optimizationBudgetExceeded)
    return std::nullopt;
  uint64_t excess = 0;
  unsigned maximum = 0;
  for (unsigned anchor : plan->groups)
    for (unsigned w = 0; w < waves; ++w)
      for (unsigned first = 0; first < 64; first += 8) {
        std::array<SmallVector<unsigned, 4>, 32> banks;
        for (unsigned l = first; l < first + 8; ++l) {
          unsigned sr = plan->matrixAlt ? ((l >> 2) & 3)
                                       : ((l >> 2) & 3) | (((l >> 1) & 1) << 2);
          unsigned sl = (l & 48) | (plan->matrixAlt ? ((l & 3) << 2)
                                                        : ((l & 1) << 3));
          unsigned r = plan->repartition ? plan->matrixByAnchor[anchor][sr]
                                         : anchor + sr;
          unsigned word = physical[(r * waves + w) * 64 + sl] / 2;
          for (unsigned j = 0; j < 4; ++j) {
            auto &bank = banks[(word + j) % 32];
            if (!llvm::is_contained(bank, word + j))
              bank.push_back(word + j);
          }
        }
        for (const auto &bank : banks)
          if (!bank.empty()) {
            excess += bank.size() - 1;
            maximum = std::max(maximum, unsigned(bank.size()));
          }
      }
  return std::make_pair(excess, maximum);
}

// Select the shared chart from the actual matrix contiguity and dot consumer.
// The ordinary LinearLayout is also the source-coordinate map for G2L; no MLS
// instruction tile, packet layout or source-side lane numbering is introduced.
inline gpu::SharedLinearEncodingAttr
getMatrixDescriptorSharedLayout(triton::MatrixLoadOp op) {
  if (!isAsyncMatrixDescriptor(op) || op.getStrides().size() != 2)
    return {};
  auto consumer = getDotOpIdxFromMatrixLoad(op);
  if (failed(consumer))
    return {};
  auto [dot, operand] = *consumer;
  auto dotType = cast<RankedTensorType>(dot->getOperand(operand).getType());
  auto encoding = dyn_cast<gpu::DotOperandEncodingAttr>(dotType.getEncoding());
  if (!encoding)
    return {};
  APInt first, second;
  bool firstUnit = matchPattern(op.getStrides()[0], m_ConstantInt(&first)) &&
                   first.isOne();
  bool secondUnit = matchPattern(op.getStrides()[1], m_ConstantInt(&second)) &&
                    second.isOne();
  if (!firstUnit && !secondUnit)
    return {};
  unsigned column = secondUnit ? 1 : 0;
  SmallVector<unsigned> order{column, 1 - column};
  auto type = op.getType();
  if (type.getRank() != 2 || type.getShape()[column] < 8)
    return {};
  auto cta = gpu::getCTALayout(type.getEncoding());
  auto consumerShared = gpu::SwizzledSharedEncodingAttr::get(
      op.getContext(), encoding, type.getShape(), order, cta,
      type.getElementTypeBitWidth(), false);
  unsigned phases = std::min<unsigned>(consumerShared.getMaxPhase(),
                                      type.getShape()[column] / 8);
  auto shared = gpu::SwizzledSharedEncodingAttr::get(
      op.getContext(), 8, consumerShared.getPerPhase(), phases, order, cta);
  auto result = gpu::SharedLinearEncodingAttr::get(
      op.getContext(), gpu::toLinearLayout(type.getShape(), shared), 16);
  auto module = op->getParentOfType<ModuleOp>();
  auto target = module->getAttrOfType<StringAttr>("ttg.target");
  if (!target || target.getValue() != "hip:gfx936" ||
      type.getElementTypeBitWidth() != 16)
    return result;

  // Propagate the consumer's ownership through pure views instead of treating
  // a reshape/transpose as a new memory protocol or trusting an operand tag.
  Attribute readEncoding = gpu::LinearEncodingAttr::get(
      op.getContext(), gpu::toLinearLayout(dotType));
  Value value = dot->getOperand(operand);
  while (value != op.getResult()) {
    auto *view = value.getDefiningOp();
    if (!view)
      return result;
    if (!isa<gpu::ConvertLayoutOp>(view)) {
      if (!isa<triton::ReshapeOp, triton::TransOp>(view))
        return result;
      readEncoding = mlir::inferSrcEncoding(view, readEncoding);
      if (!readEncoding)
        return result;
    }
    value = view->getOperand(0);
  }
  auto readType = type.cloneWithEncoding(readEncoding);
  auto baselineCost = getDescriptorReadBankCost(readType, result);
  if (!baselineCost)
    return result;
  auto bestCost = *baselineCost;
  unsigned bestPhases = 0;
  // All candidates retain the actual16B G2L vectors. Explore only ordinary
  // XOR swizzles over one32-bank/four-byte bank span; keep the established
  // layout on ties and prefer fewer XORs among strict bank improvements.
  unsigned bankElements = 32 * 4 / 2;
  for (unsigned vec = 8; vec < bankElements; vec *= 2) {
    unsigned maxPhase = std::min<unsigned>(bankElements / vec,
        std::min<int64_t>(type.getShape()[column] / vec,
                          type.getShape()[1 - column]));
    if (!maxPhase)
      continue;
    auto candidate = gpu::SwizzledSharedEncodingAttr::get(
        op.getContext(), vec, 1, maxPhase, order, cta);
    auto linear = gpu::SharedLinearEncodingAttr::get(
        op.getContext(), gpu::toLinearLayout(type.getShape(), candidate), 16);
    auto cost = getDescriptorReadBankCost(readType, linear);
    if (cost && (*cost < bestCost ||
        (*cost == bestCost && bestCost < *baselineCost &&
         maxPhase < bestPhases))) {
      result = linear;
      bestCost = *cost;
      bestPhases = maxPhase;
    }
  }
  return result;
}
} // namespace mlir::triton::HCU
#endif
