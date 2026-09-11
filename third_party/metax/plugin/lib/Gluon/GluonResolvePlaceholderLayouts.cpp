#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"
#include "Gluon/Passes.h"

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Gluon/Transforms/InferLayoutUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "metax-gluon-resolve-placeholder-layouts"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gd = ::mlir::triton::gluon;
namespace layout = ::mlir::triton::gpu::metax::gluon;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONRESOLVEPLACEHOLDERLAYOUTS
#include "Gluon/Passes.h.inc"

namespace {

bool isAutoTensorType(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isa<gd::AutoEncodingAttr>(tensor.getEncoding());
}

LogicalResult unwrapNoVerifyLayouts(ModuleOp module) {
  unsigned count = layout::unwrapNoVerifyEncodings(module);
  LDBG("unwrapped " << count << " no-verify encoding attribute(s)");
  for (tt::FuncOp func : module.getOps<tt::FuncOp>())
    layout::synchronizeFunctionTypeAfterNoVerifyUnwrap(func);
  return layout::verifyNoResidualNoVerifyEncodings(
      module, "MetaX Gluon placeholder resolution");
}

Attribute makeDefaultBlockedEncoding(Value value, ModuleOp module) {
  auto type = cast<RankedTensorType>(value.getType());
  unsigned rank = type.getRank();
  SmallVector<unsigned> sizePerThread(rank, 1);
  SmallVector<unsigned> order(rank);
  std::iota(order.rbegin(), order.rend(), 0);

  Operation *context = value.getDefiningOp();
  if (!context)
    context = cast<BlockArgument>(value).getOwner()->getParentOp();
  int numWarps = ttg::lookupNumWarps(context ? context : module);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(module);
  return ttg::BlockedEncodingAttr::get(
      module.getContext(), type.getShape(), sizePerThread, order, numWarps,
      threadsPerWarp, numCTAs);
}

SmallVector<Value> collectAutoTensorValues(tt::FuncOp func) {
  SmallVector<Value> values;
  DenseSet<Value> seen;
  func.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (isAutoTensorType(argument.getType()) &&
              seen.insert(argument).second)
            values.push_back(argument);
    for (Value result : op->getResults())
      if (isAutoTensorType(result.getType()) && seen.insert(result).second)
        values.push_back(result);
  });
  return values;
}

SmallVector<Value> findDeterministicRoots(tt::FuncOp func,
                                          ArrayRef<Value> values) {
  layout::GluonRegionBranchAnalysis carriers(func);
  (void)carriers.initialize();

  SmallVector<Value> roots;
  for (Value value : values) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (carriers.getPredecessors(argument).empty())
        roots.push_back(value);
      continue;
    }

    // Prefer downstream boundaries. Rank-changing operations infer their
    // source as a SliceEncoding from a concrete result; choosing an upstream
    // make_range first would instead invent a rank-1 Blocked root that cannot
    // legally feed expand_dims.
    bool hasAutoSuccessor = false;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (isa<RegionBranchOpInterface,
              RegionBranchTerminatorOpInterface>(user) ||
          llvm::any_of(user->getResults(), [](Value result) {
            return isAutoTensorType(result.getType());
          })) {
        hasAutoSuccessor = true;
        break;
      }
    }
    if (!hasAutoSuccessor)
      roots.push_back(value);
  }
  if (roots.empty() && !values.empty())
    roots.push_back(values.back());
  return roots;
}

void eraseIdentityLayoutConversions(tt::FuncOp func) {
  SmallVector<ttg::ConvertLayoutOp> identity;
  func.walk([&](ttg::ConvertLayoutOp convert) {
    if (convert.getSrc().getType() == convert.getType())
      identity.push_back(convert);
  });
  for (ttg::ConvertLayoutOp convert : identity) {
    convert.replaceAllUsesWith(convert.getSrc());
    convert.erase();
  }
}

LogicalResult resolveGenericAutoTensors(tt::FuncOp func, ModuleOp module) {
  while (true) {
    SmallVector<Value> values = collectAutoTensorValues(func);
    if (values.empty())
      return success();

    SmallVector<Value> roots = findDeterministicRoots(func, values);
    if (roots.empty())
      return func.emitError()
             << "cannot find a deterministic root for residual Auto tensors";

    // Resolve one connected placeholder component at a time. Seeding every
    // syntactic root in one call is incorrect when independent rank-changing
    // branches share a make_range or splat ancestor: each root has a valid
    // default Blocked layout, but those defaults need a local conversion
    // boundary rather than a global consensus.
    Value root = roots.front();
    SmallVector<std::pair<Value, Attribute>> seed{
        {root, makeDefaultBlockedEncoding(root, module)}};
    LDBG("resolving one generic Auto component rooted at " << root);
    if (failed(gd::inferLayout(func, isAutoTensorType, seed)))
      return failure();

    if (collectAutoTensorValues(func).size() >= values.size())
      return func.emitError()
             << "generic Auto tensor completion made no progress";
  }
}

LogicalResult resolvePlaceholders(ModuleOp module) {
  if (failed(layout::verifyNoResidualCalls(
          module, "MetaX Gluon placeholder resolution")) ||
      failed(unwrapNoVerifyLayouts(module)))
    return failure();

  for (tt::FuncOp func : module.getOps<tt::FuncOp>())
    if (failed(resolveGenericAutoTensors(func, module)))
      return failure();
    else
      eraseIdentityLayoutConversions(func);

  return failure(
      failed(layout::verifyNoResidualAutoEncodings(
          module, "MetaX Gluon placeholder resolution")) ||
      failed(layout::verifyNoResidualNoVerifyEncodings(
          module, "MetaX Gluon placeholder resolution")));
}

class TritonMETAXGPUGluonResolvePlaceholderLayoutsPass
    : public impl::TritonMETAXGPUGluonResolvePlaceholderLayoutsBase<
          TritonMETAXGPUGluonResolvePlaceholderLayoutsPass> {
public:
  void runOnOperation() override {
    if (failed(resolvePlaceholders(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
createTritonMETAXGPUGluonResolvePlaceholderLayoutsPass() {
  return std::make_unique<
      TritonMETAXGPUGluonResolvePlaceholderLayoutsPass>();
}

} // namespace mlir

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
