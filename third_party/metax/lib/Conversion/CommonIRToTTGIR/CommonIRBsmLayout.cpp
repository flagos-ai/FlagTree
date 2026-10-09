// MIT License
// Copyright (c) 2026 The FlagOS Contributors

#include "triton/Conversion/CommonIRToTTGIR/BsmLayout.h"
#include "triton/Conversion/CommonIRToTTGIR/LayoutRules.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <cstdlib>

namespace mlir::triton::metax {

namespace ttg = mlir::triton::gpu;

FailureOr<std::optional<BsmLayoutChain>> getBsmLayoutChain(Value operand) {
  BsmLayoutChain chain;
  auto skipConversions = [&](Value value) {
    while (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      chain.conversions.push_back(cvt);
      value = cvt.getSrc();
    }
    return value;
  };
  Value value = skipConversions(operand);
  chain.perm = value.getDefiningOp<ttg::BsmPermOp>();
  if (!chain.perm)
    return std::optional<BsmLayoutChain>();
  chain.load =
      skipConversions(chain.perm.getSrc1()).getDefiningOp<ttg::LocalLoadOp>();
  if (!chain.load || chain.load.getMmaMode() != 2) {
    chain.perm.emitError("BSM requires a local_load with mmaMode=2");
    return failure();
  }
  return std::optional<BsmLayoutChain>(std::move(chain));
}

namespace {

constexpr unsigned kBsmSlots = 32;

// Each BSM output selects one 16-bit half of an i32 carrier slot. This is the
// same index/coverage check as Gluon's analyzeBsmPermPhysicalContract,
// retaining the mapping so the fixed-address LDS implementation can also be
// checked.
using BsmMapping = std::array<std::pair<unsigned, unsigned>, kBsmSlots>;
std::optional<BsmMapping> getBsmMapping(unsigned n, unsigned k) {
  if (!n || !k || n % 2 || k % 2 || uint64_t(n) * k != kBsmSlots)
    return std::nullopt;
  BsmMapping mapping;
  std::array<unsigned, kBsmSlots> writes{};
  for (unsigned j = 0; j < n * k / 2; j += 2 * n) {
    for (unsigned v = 0; v < n / 2; ++v) {
      unsigned source = (j / n + v) * k;
      unsigned output = 2 * v + n * (j / k);
      for (unsigned upper : {0u, 1u}) {
        for (unsigned half : {0u, 1u}) {
          for (unsigned second : {0u, 1u}) {
            unsigned src = source + upper * n + second;
            unsigned dst = output + (2 * upper + half) * k + second;
            if (src >= kBsmSlots || dst >= kBsmSlots || ++writes[dst] != 1)
              return std::nullopt;
            mapping[dst] = {src, half};
          }
        }
      }
    }
  }
  if (llvm::any_of(writes, [](unsigned count) { return count != 1; }))
    return std::nullopt;
  return mapping;
}

// Mirror the address-only part of lowerLocalLdswithoutPerm. In addition to
// avoiding its fallback/assertion paths, prove that its hard-coded v2i32 loads
// and +512-element offset produce the selected logical B layout after BSM.
bool checkSplitLoadAddresses(LinearLayout cvt, ttg::MemDescType shared,
                             unsigned elementsN, const BsmMapping &mapping) {
  auto *ctx = shared.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto block = StringAttr::get(ctx, "block");
  auto offset = StringAttr::get(ctx, "offset");
  if (!cvt.isTrivialOver({block}))
    return false;
  cvt = cvt.sublayout({reg, lane, warp}, {offset});
  if (!actionRemoveBroadcastedRegs(cvt).isIdentity())
    return false;

  unsigned vec = std::min(elementsN, 8u); // 128-bit maximum / 16-bit elements
  // The native implementation always emits v2i32, i.e. four 16-bit elements.
  if (vec != 4)
    return false;
  int32_t columns = shared.getShape()[ttg::getOrder(shared)[0]];
  std::vector<std::vector<int32_t>> tileBases = {{columns}, {1}, {2}};
  LinearLayout tile({{reg, tileBases}}, {{offset, llvm::NextPowerOf2(columns)}},
                    false);
  auto permutation = regPermForDivide(cvt, tile, /*left=*/true);
  if (!permutation)
    return false;
  auto permuted = permutation->apply(cvt);
  auto bases = permuted.getBases();
  auto &registerBases = bases[reg];
  for (unsigned i = 0; i < tileBases.size(); ++i) {
    if (registerBases[i] != tileBases[i])
      return false;
    registerBases[i] = {0};
  }
  for (auto dim : cvt.getInDimNames())
    for (auto basis : bases[dim])
      for (auto tileBasis : tileBases)
        if (basis[0] & tileBasis[0])
          return false;
  LinearLayout reps(bases, permuted.getOutDims(), false);
  LinearLayout addresses({{lane, bases[lane]}, {warp, bases[warp]}},
                         reps.getOutDims(), false);
  // Full buffers and slot views have no offset bits within this rank-2 shape.
  auto [additive, strides] = actionAdditiveStrides(reps, addresses, 0);
  if (!strides.isIdentity() || additive < 4 * vec || additive > kBsmSlots ||
      kBsmSlots % additive)
    return false;

  for (int w = 0; w < cvt.getInDimSize(warp); ++w) {
    for (int l = 0; l < cvt.getInDimSize(lane); ++l) {
      std::array<int32_t, kBsmSlots> raw;
      raw.fill(-1);
      int32_t base = addresses.apply({{lane, l}, {warp, w}})[0].second;
      for (unsigned i = 0; i < kBsmSlots; i += additive) {
        int32_t outer =
            base ^ reps.apply({{reg, i}, {lane, 0}, {warp, 0}})[0].second;
        for (unsigned j = 0; j < additive / 2; j += 2 * vec) {
          int32_t inner =
              outer + reps.apply({{reg, j}, {lane, 0}, {warp, 0}})[0].second;
          for (unsigned v = 0; v < vec / 2; ++v) {
            for (unsigned upper : {0u, 1u}) {
              for (unsigned duplicate : {0u, 1u}) {
                for (unsigned second : {0u, 1u}) {
                  unsigned index = i + j + 4 * v + 2 * duplicate + second +
                                   upper * (kBsmSlots / 2);
                  if (index >= kBsmSlots || raw[index] != -1)
                    return false;
                  raw[index] = inner + 2 * v + second * columns + upper * 512;
                }
              }
            }
          }
        }
      }
      for (unsigned r = 0; r < kBsmSlots; ++r) {
        auto [source, half] = mapping[r];
        int32_t expected =
            cvt.apply({{reg, r}, {lane, l}, {warp, w}})[0].second;
        if (raw[source] < 0 || raw[source] + half != expected)
          return false;
      }
    }
  }
  return true;
}

} // namespace

std::optional<std::string> checkBsmLayout(BsmLayoutChain chain,
                                          Attribute encoding) {
  auto dot = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(encoding);
  auto mma =
      dot ? dyn_cast<ttg::MACAMmaEncodingAttr>(dot.getParent()) : nullptr;
  if (!mma || dot.getOpIdx() != 1 || mma.getVersionMajor() != 2 ||
      mma.getIsATrans() || mma.getIsBTrans())
    return "BSM requires a non-transposed MACA v2 B operand";
  if (std::getenv("TRITON_DISABLE_SPLIT_PERM"))
    return "BSM split load is disabled by TRITON_DISABLE_SPLIT_PERM";
  auto raw = cast<RankedTensorType>(chain.load.getType());
  auto result = cast<RankedTensorType>(chain.perm.getType());
  auto shared = cast<ttg::MemDescType>(chain.load.getSrc().getType());
  if (raw.getRank() != 2 || result.getRank() != 2 || shared.getRank() != 2 ||
      raw.getShape() != result.getShape() ||
      raw.getShape() != shared.getShape() ||
      !raw.getElementType().isInteger(32) ||
      (!result.getElementType().isF16() && !result.getElementType().isBF16()) ||
      result.getElementType() != shared.getElementType())
    return "BSM requires matching rank-2 f16/bf16 shared storage, i32 load "
           "and f16/bf16 result";
  if (!isa<ttg::SwizzledSharedEncodingAttr>(shared.getEncoding()) ||
      shared.getAllocShape().take_back(2) != shared.getShape() ||
      chain.load.getSrc().getDefiningOp<ttg::MemDescTransOp>())
    return "BSM requires an untransposed swizzled shared buffer or slot view";
  if (mma.getRepOrderForOperand(1) == ttg::getOrder(shared))
    return "BSM requires shared order different from the MMA B register order";
  auto elements = mma.getElementsMNK();
  if (elements.size() != 3)
    return "BSM requires rank-2 MMA elementsMNK";
  auto mapping = getBsmMapping(elements[1], elements[2]);
  if (!mapping)
    return "BSM permutation must cover its 32 element slots exactly once";
  auto rawLayout = ttg::toLinearLayout(raw.cloneWithEncoding(encoding));
  auto resultLayout = ttg::toLinearLayout(result.cloneWithEncoding(encoding));
  auto reg = StringAttr::get(raw.getContext(), "register");
  if (rawLayout.getInDimSize(reg) != kBsmSlots ||
      resultLayout.getInDimSize(reg) != kBsmSlots)
    return "BSM source and result must each have 32 per-thread element slots";
  if (!checkSplitLoadAddresses(
          rawLayout.invertAndCompose(ttg::toLinearLayout(shared)), shared,
          elements[1], *mapping))
    return "split LDS addresses do not implement the selected BSM layout";
  return std::nullopt;
}

LogicalResult
collectBsmLayoutRequirements(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                             tile::TensorLayoutRequirements &requirements) {
  llvm::MapVector<Value, Attribute> assignments;
  llvm::SetVector<OpOperand *> coveredUses;
  for (auto [dot, plan] : choices) {
    auto found = getBsmLayoutChain(dot.getB());
    if (failed(found))
      return failure();
    if (!*found)
      continue;
    auto chain = **found;
    Attribute encoding = plan.operandB;
    // Both selected candidates and existing MMA layouts are validated here.
    if (auto error = checkBsmLayout(chain, encoding))
      return chain.perm.emitError(*error);
    auto requireLayout = [&](Value value) -> LogicalResult {
      auto [it, inserted] = assignments.insert({value, encoding});
      if (inserted || it->second == encoding)
        return success();
      return dot.emitError(
                 "shared split-load/BSM value requires conflicting layouts: ")
             << it->second << " versus " << encoding;
    };
    for (Value value : {chain.load.getResult(), chain.perm.getResult()})
      if (failed(requireLayout(value)))
        return failure();
    for (auto cvt : chain.conversions) {
      if (failed(requireLayout(cvt.getResult())))
        return failure();
      coveredUses.insert(&cvt->getOpOperand(0));
    }
    coveredUses.insert(&chain.perm->getOpOperand(0));
    coveredUses.insert(&dot->getOpOperand(1));
  }
  auto result = module.walk([&](Operation *op) {
    auto load = dyn_cast<ttg::LocalLoadOp>(op);
    if ((isa<ttg::BsmPermOp>(op) || (load && load.getMmaMode() == 2)) &&
        !assignments.contains(op->getResult(0))) {
      op->emitError(
          "split load/BSM must feed dot B through layout conversions only");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  // Packed raw values cannot be handed to arbitrary users via an ordinary
  // convert_layout. Validate all uses before publishing any requirements.
  for (auto [value, encoding] : assignments) {
    for (OpOperand &use : value.getUses()) {
      if (!coveredUses.contains(&use))
        return use.getOwner()->emitError(
            "BSM layout assignment does not support this use outside "
            "split load/BSM/dot B chains");
    }
  }
  for (auto [value, encoding] : assignments)
    requirements.assignments[value] = encoding;
  for (OpOperand *use : coveredUses)
    requirements.uses.push_back({use, assignments.lookup(use->get()), false});
  return success();
}

} // namespace mlir::triton::metax
