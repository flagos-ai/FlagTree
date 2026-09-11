#include "Gluon/GluonRegisterToShared.h"

#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/Targets/GluonC500Layout.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "metax-gluon-register-to-shared"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;

namespace mlir::triton::gpu::metax::gluon {
namespace {

enum class BoundaryRequirement {
  Lowerable,
  FullCoverageWithoutWarpBroadcast,
};

static bool satisfies(const RegisterToSharedContractInfo &contract,
                      ttg::MemDescType shared,
                      BoundaryRequirement requirement) {
  if (!contract.lowerable)
    return false;
  if (requirement == BoundaryRequirement::Lowerable)
    return true;
  bool hasPadding = isa<ttg::PaddedSharedEncodingAttr>(
      unwrapNoVerifyEncoding(shared.getEncoding()));
  return contract.freeWarpMask == 0 &&
         (hasPadding || contract.completeCoverage);
}

static std::optional<SmallVector<unsigned>>
getSharedOrder(ttg::MemDescType type) {
  Attribute encoding = unwrapNoVerifyEncoding(type.getEncoding());
  if (auto shared = dyn_cast_or_null<ttg::SharedEncodingTrait>(encoding))
    return ttg::getOrder(shared, type.getShape());
  if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(encoding))
    return SmallVector<unsigned>(blocked.getOrder());
  return std::nullopt;
}

static FailureOr<RankedTensorType>
inferBoundaryType(RankedTensorType tensor, ttg::MemDescType shared,
                  BoundaryRequirement requirement, Operation *anchor) {
  if (!tensor || !shared || !anchor ||
      tensor.getShape() != shared.getShape() ||
      tensor.getElementType() != shared.getElementType())
    return failure();
  auto accepts = [&](Attribute encoding) {
    FailureOr<RegisterToSharedContractInfo> contract =
        checkRegisterToSharedContract(tensor.cloneWithEncoding(encoding),
                                      shared);
    return succeeded(contract) &&
           satisfies(*contract, shared, requirement);
  };

  Attribute current = unwrapNoVerifyEncoding(tensor.getEncoding());
  if (current && accepts(current))
    return tensor.cloneWithEncoding(current);

  std::optional<SmallVector<unsigned>> order = getSharedOrder(shared);
  unsigned elementBits = c500::getTensorElementOrPointeeBitWidth(tensor);
  if (!order || order->empty() || !elementBits || elementBits > 128)
    return failure();
  unsigned contiguousDimension = order->front();
  int64_t extent = tensor.getShape()[contiguousDimension];
  if (extent <= 0)
    return failure();
  unsigned maximum = std::min<unsigned>(
      static_cast<unsigned>(extent), 128 / elementBits);
  if (auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(
          unwrapNoVerifyEncoding(shared.getEncoding())))
    maximum = std::min(maximum, padded.getMinInterval());
  int numWarps = ttg::lookupNumWarps(anchor);
  OpBuilder builder(anchor);
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  if (numWarps <= 0 || threadsPerWarp <= 0)
    return failure();

  for (unsigned elements =
           static_cast<unsigned>(llvm::bit_floor(maximum));
       elements != 0; elements >>= 1) {
    SmallVector<unsigned> sizePerThread(tensor.getRank(), 1);
    sizePerThread[contiguousDimension] = elements;
    Attribute encoding = ttg::BlockedEncodingAttr::get(
        tensor.getContext(), tensor.getShape(), sizePerThread, *order,
        static_cast<unsigned>(numWarps),
        static_cast<unsigned>(threadsPerWarp),
        ttg::getCTALayout(shared.getEncoding()));
    if (accepts(encoding))
      return tensor.cloneWithEncoding(encoding);
  }
  return failure();
}

static bool hasMemDescTranspose(Value value) {
  DenseSet<Value> visited;
  while (value && visited.insert(value).second) {
    Operation *def = value.getDefiningOp();
    if (isa_and_nonnull<ttg::MemDescTransOp>(def))
      return true;
    value = c500::getMemDescViewSource(def);
  }
  return false;
}

static LogicalResult legalizeRegisterSharedWrites(tt::FuncOp func) {
  SmallVector<std::pair<Operation *, unsigned>, 8> sinks;
  func.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<ttg::LocalStoreOp>(
            [&](ttg::LocalStoreOp) { sinks.emplace_back(op, 0); })
        .Case<ttg::LocalAllocOp>([&](ttg::LocalAllocOp alloc) {
          if (alloc.getSrc())
            sinks.emplace_back(op, 0);
        });
  });

  for (auto [owner, operandIndex] : sinks) {
    Value source = owner->getOperand(operandIndex);
    auto tensor = dyn_cast<RankedTensorType>(source.getType());
    auto shared =
        dyn_cast<ttg::LocalStoreOp>(owner)
            ? dyn_cast<ttg::MemDescType>(
                  cast<ttg::LocalStoreOp>(owner).getDst().getType())
            : dyn_cast<ttg::MemDescType>(
                  cast<ttg::LocalAllocOp>(owner).getType());
    if (!tensor || !shared)
      continue;
    FailureOr<RankedTensorType> boundary =
        inferBoundaryType(tensor, shared, BoundaryRequirement::Lowerable,
                          owner);
    if (failed(boundary))
      return owner->emitError()
             << "C500 cannot form a legal register-to-shared mapping";
    if (*boundary == tensor)
      continue;
    OpBuilder builder(owner);
    Value converted = ttg::ConvertLayoutOp::create(
        builder, owner->getLoc(), *boundary, source);
    owner->setOperand(operandIndex, converted);
    LDBG("[register-shared-write] source="
         << tensor.getEncoding() << ", boundary="
         << boundary->getEncoding() << ", shared="
         << shared.getEncoding());
  }
  return success();
}

static LogicalResult legalizeDotLoads(tt::FuncOp func) {
  SmallVector<ttg::LocalLoadOp, 8> loads;
  func.walk([&](ttg::LocalLoadOp load) {
    if (isSimpleDotLocalLoad(load) && load.getMmaMode() == -1)
      loads.push_back(load);
  });

  for (ttg::LocalLoadOp load : loads) {
    auto tensor = dyn_cast<RankedTensorType>(load.getType());
    auto shared = dyn_cast<ttg::MemDescType>(load.getSrc().getType());
    if (!tensor || !shared)
      continue;
    FailureOr<RegisterToSharedContractInfo> direct =
        checkRegisterToSharedContract(tensor, shared);
    bool needsBoundary =
        failed(direct) || !direct->lowerable ||
        (hasMemDescTranspose(load.getSrc()) &&
         direct->freeWarpMask != 0);
    if (!needsBoundary)
      continue;
    FailureOr<RankedTensorType> boundary = inferBoundaryType(
        tensor, shared,
        BoundaryRequirement::FullCoverageWithoutWarpBroadcast, load);
    if (failed(boundary))
      return load.emitError()
             << "C500 cannot legalize the shared-to-dot mapping";
    OpBuilder builder(load);
    auto blockedLoad = ttg::LocalLoadOp::create(
        builder, load.getLoc(), *boundary, load.getSrc(), load.getToken());
    blockedLoad->setAttrs(load->getAttrs());
    Value dotValue = ttg::ConvertLayoutOp::create(
        builder, load.getLoc(), tensor, blockedLoad);
    load.replaceAllUsesWith(dotValue);
    load.erase();
    LDBG("[shared-dot-boundary] shared="
         << shared.getEncoding() << ", blocked="
         << boundary->getEncoding() << ", dot=" << tensor.getEncoding());
  }
  return success();
}

} // namespace

LogicalResult legalizeLocalDotStaging(ModuleOp module) {
  for (tt::FuncOp func : module.getOps<tt::FuncOp>())
    if (failed(legalizeRegisterSharedWrites(func)) ||
        failed(legalizeDotLoads(func)))
      return failure();
  return success();
}

} // namespace mlir::triton::gpu::metax::gluon

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
