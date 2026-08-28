#ifdef __TLE__

#include "MUSATLE/Transforms/PipePartitionUtils.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace mlir::triton::musa_tle {

namespace ttg = triton::gpu;

namespace {

constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";

static LogicalResult verifyMarkedWarpSpecialize(ttg::WarpSpecializeOp ws,
                                                ModuleOp module) {
  SmallVector<ttg::WarpSpecializeOp> marked;
  module.walk([&](ttg::WarpSpecializeOp candidate) {
    if (candidate->hasAttr(kStaticWarpSpecializeAttr))
      marked.push_back(candidate);
  });
  if (marked.size() != 1 || marked.front() != ws)
    return ws.emitOpError(
        "MUSA TLE pipe requires one marked static warp_specialize owner");

  auto func = ws->getParentOfType<::mlir::triton::FuncOp>();
  if (!func || ws->getBlock() != &func.getBody().front())
    return ws.emitOpError(
        "MUSA TLE pipe requires a top-level static warp_specialize");
  return success();
}

static FailureOr<int32_t> getPositiveModuleAttr(ModuleOp module, StringRef name,
                                                Operation *anchor) {
  auto attr = module->getAttrOfType<IntegerAttr>(name);
  if (!attr || attr.getInt() <= 0 ||
      attr.getInt() > std::numeric_limits<int32_t>::max())
    return anchor->emitOpError()
           << "MUSA TLE static warp partition requires positive " << name;
  return static_cast<int32_t>(attr.getInt());
}

static FailureOr<SmallVector<int32_t>> getWorkerStarts(ttg::WarpSpecializeOp ws,
                                                       int32_t defaultWarps,
                                                       Operation *anchor) {
  ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
  if (workerWarps.empty())
    return anchor->emitOpError(
        "MUSA TLE static warp partition requires at least one worker");

  SmallVector<int32_t> starts;
  starts.reserve(workerWarps.size());
  int64_t next = defaultWarps;
  for (auto [index, count] : llvm::enumerate(workerWarps)) {
    if (count <= 0)
      return anchor->emitOpError()
             << "MUSA TLE static worker partition #" << index
             << " requires a positive warp count";
    if (next <= 0 || next > std::numeric_limits<int32_t>::max())
      return anchor->emitOpError(
          "MUSA TLE static warp partition warp range exceeds i32");
    starts.push_back(static_cast<int32_t>(next));
    next += count;
    if (next > std::numeric_limits<int32_t>::max())
      return anchor->emitOpError(
          "MUSA TLE static warp partition warp range exceeds i32");
  }

  if (auto existing = ws.getWarpGroupStartIds()) {
    if (existing->size() != starts.size() || !llvm::equal(*existing, starts))
      return anchor->emitOpError(
          "MUSA TLE static worker warp ranges must follow declaration order");
  }

  if (auto total = ws->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
          "ttg.total-num-warps")) {
    if (total.getInt() != next)
      return anchor->emitOpError(
          "MUSA TLE static worker warp ranges conflict with total warps");
  }
  return starts;
}

} // namespace

FailureOr<std::optional<PipeStaticPartitionInfo>>
resolvePipeStaticPartition(Operation *op) {
  if (!op)
    return failure();

  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op->emitOpError(
        "MUSA TLE pipe operation must be nested in a module");

  ttg::WarpSpecializeOp owner;
  Region *partitionRegion = nullptr;
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto candidate = dyn_cast<ttg::WarpSpecializeOp>(parent)) {
      if (region == &candidate.getDefaultRegion()) {
        owner = candidate;
        partitionRegion = region;
        break;
      }
    }
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
      auto candidate =
          dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
      if (!candidate)
        return op->emitOpError("MUSA TLE pipe operation is not in a recognized "
                               "static warp-specialize partition");
      owner = candidate;
      partitionRegion = region;
      break;
    }
    region = parent->getParentRegion();
  }

  if (!owner) {
    FailureOr<int32_t> warpCount =
        getPositiveModuleAttr(module, ttg::AttrNumWarpsName, op);
    if (failed(warpCount))
      return failure();
    return std::optional<PipeStaticPartitionInfo>(PipeStaticPartitionInfo{
        {}, nullptr, PipePartitionKind::CTA, std::nullopt, 0, 0, *warpCount});
  }

  if (failed(verifyMarkedWarpSpecialize(owner, module)))
    return failure();

  FailureOr<int32_t> defaultWarps =
      getPositiveModuleAttr(module, ttg::AttrNumWarpsName, op);
  if (failed(defaultWarps))
    return failure();
  FailureOr<SmallVector<int32_t>> workerStarts =
      getWorkerStarts(owner, *defaultWarps, op);
  if (failed(workerStarts))
    return failure();

  if (partitionRegion == &owner.getDefaultRegion()) {
    return std::optional<PipeStaticPartitionInfo>(PipeStaticPartitionInfo{
        owner, partitionRegion, PipePartitionKind::WarpSpecializeDefault,
        std::nullopt, 0, 0, *defaultWarps});
  }

  auto workerRegions = owner.getPartitionRegions();
  auto it = llvm::find(workerRegions, partitionRegion);
  if (it == workerRegions.end())
    return op->emitOpError("MUSA TLE pipe operation is not in a direct static "
                           "warp-specialize partition");
  unsigned worker = static_cast<unsigned>(it - workerRegions.begin());
  int32_t count = owner.getPartitionNumWarps()[worker];
  return std::optional<PipeStaticPartitionInfo>(PipeStaticPartitionInfo{
      owner, partitionRegion, PipePartitionKind::WarpSpecializeWorker, worker,
      worker + 1, (*workerStarts)[worker], count});
}

FailureOr<int32_t> getPipePartitionStartThread(Operation *op) {
  FailureOr<std::optional<PipeStaticPartitionInfo>> placement =
      resolvePipeStaticPartition(op);
  if (failed(placement))
    return failure();
  if (!*placement || (*placement)->kind == PipePartitionKind::CTA ||
      (*placement)->kind == PipePartitionKind::WarpSpecializeDefault)
    return 0;

  ModuleOp module = op->getParentOfType<ModuleOp>();
  int32_t threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  if (threadsPerWarp <= 0)
    return op->emitOpError(
        "MUSA TLE static warp partition requires positive threads per warp");
  int64_t start = static_cast<int64_t>((*placement)->warpBegin) *
                  static_cast<int64_t>(threadsPerWarp);
  if (start < 0 || start > std::numeric_limits<int32_t>::max())
    return op->emitOpError(
        "MUSA TLE static warp partition start thread exceeds i32 range");
  return static_cast<int32_t>(start);
}

bool samePipeStaticPartition(const PipeStaticPartitionInfo &lhs,
                             const PipeStaticPartitionInfo &rhs) {
  return lhs.owner == rhs.owner && lhs.partitionIndex == rhs.partitionIndex &&
         lhs.kind == rhs.kind && lhs.workerIndex == rhs.workerIndex &&
         lhs.warpBegin == rhs.warpBegin && lhs.warpCount == rhs.warpCount;
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
