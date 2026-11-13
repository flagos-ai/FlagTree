// gfx936 wave-private staging: coalesce redundant waits within proven read runs.
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "triton/Analysis/PaddedSlotGeometry.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace {
class WavePrivateSyncStaging {
  using Id = mlir::Allocation::BufferId;
  using MemDesc = mlir::triton::gpu::MemDescType;
  struct Provenance {
    mlir::Value root;
    MemDesc plane;
  };
  mlir::ModuleOp mod;
  mlir::Allocation *allocation = nullptr;
  llvm::DenseMap<mlir::Operation *, Id> accesses;
  llvm::DenseMap<Id, mlir::Value> roots;
  llvm::DenseMap<Id, MemDesc> canonicalTypes;
  llvm::DenseSet<Id> rejected;
  llvm::DenseSet<Id> paddedIds;
  const mlir::triton::HCU::TargetInfo &target;

  // Same batch owns the same wave for every register/lane. Higher batch
  // bits may live in registers; no lane or register can change low wave bits.
  bool ownedByBatchWave(mlir::RankedTensorType ty) const {
    using namespace mlir;
    if (ty.getRank() != 3)
      return false;
    auto layout = triton::gpu::toLinearLayout(ty);
    auto *ctx = mod->getContext();
    auto reg = StringAttr::get(ctx, "register");
    auto lane = StringAttr::get(ctx, "lane");
    auto warp = StringAttr::get(ctx, "warp");
    auto block = StringAttr::get(ctx, "block");
    auto bases = layout.getBases();
    auto warpBases = bases.lookup(warp);
    auto warpAttr = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
    unsigned warps = warpAttr ? warpAttr.getInt() : 0;
    if (warps <= 1 || !llvm::isPowerOf2_32(warps) ||
        ty.getShape()[0] < warps ||
        warpBases.size() != llvm::Log2_32(warps) ||
        !bases.lookup(block).empty())
      return false;
    for (auto [i, basis] : llvm::enumerate(warpBases))
      if (basis != std::vector<int32_t>({int32_t(1u << i), 0, 0}))
        return false;
    for (const auto &basis : bases.lookup(lane))
      if (basis[0] != 0)
        return false;
    for (const auto &basis : bases.lookup(reg))
      if ((unsigned(basis[0]) & (warps - 1)) != 0)
        return false;
    return true;
  }

  // Restrict forwarding to identical SSA incoming values, not just equal
  // descriptor types or an allocation-analysis may-alias set. Cycles decline.
  mlir::Value identicalIncoming(mlir::Value value) const {
    using namespace mlir;
    SmallVector<Value, 8> seen;
    for (unsigned depth = 0; depth < 8; ++depth) {
      auto arg = dyn_cast<BlockArgument>(value);
      if (!arg)
        return value;
      if (llvm::is_contained(seen, value))
        return {};
      seen.push_back(value);
      Block *block = arg.getOwner();
      Value incoming;
      for (Block *pred : block->getPredecessors()) {
        auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
        if (!branch)
          return {};
        for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
          if (branch->getSuccessor(i) != block)
            continue;
          auto operands = branch.getSuccessorOperands(i);
          unsigned produced = operands.getProducedOperandCount();
          unsigned number = arg.getArgNumber();
          if (number < produced ||
              number - produced >= operands.getForwardedOperands().size())
            return {};
          Value next = operands.getForwardedOperands()[number - produced];
          if (incoming && incoming != next)
            return {};
          incoming = next;
        }
      }
      if (!incoming)
        return {};
      value = incoming;
    }
    return {};
  }

  // Rank3 full allocation, exact padded physical batch planes. This uses the
  // SAME linear component and additive padding map as lowerLocalLdSt. It does
  // not estimate plane stride as allocationBytes / batchCount.
  bool paddedPlanes(MemDesc ty, Id id) const {
    using namespace mlir;
    namespace ttg = triton::gpu;
    auto enc = dyn_cast<ttg::PaddedSharedEncodingAttr>(ty.getEncoding());
    if (!enc || ty.getRank() != 3 || ty.getShape() != ty.getAllocShape() ||
        ArrayRef<int64_t>(ttg::getAllocationShapePerCTA(ty)) != ty.getShape() ||
        !(ty.getElementType().isF16() || ty.getElementType().isBF16()) ||
        !isa<ttg::SharedMemorySpaceAttr>(ty.getMemorySpace()))
      return false;
    uint64_t elements = 1;
    for (int64_t size : ty.getShape())
      if (size <= 0 || !llvm::isPowerOf2_64(size) ||
          !ttg::paddedSlotMul(elements, uint64_t(size), elements))
        return false;
    if (elements > uint64_t(std::numeric_limits<int32_t>::max()))
      return false;
    uint64_t planeElements;
    if (!ttg::paddedSlotMul(uint64_t(ty.getShape()[1]),
                            uint64_t(ty.getShape()[2]), planeElements))
      return false;
    const auto &ll = enc.getLinearComponent();
    auto offset = StringAttr::get(ty.getContext(), "offset");
    auto block = StringAttr::get(ty.getContext(), "block");
    bool hasOffset = false, hasBlock = false;
    for (StringAttr name : ll.getInDimNames()) {
      if (name == offset)
        hasOffset = true;
      else if (name == block)
        hasBlock = true;
      else
        return false;
    }
    if (!hasOffset || !hasBlock || !ll.isInvertible() ||
        ll.getNumOutDims() != 3 || ll.getInDimSize(block) != 1 ||
        uint64_t(ll.getInDimSize(offset)) != elements)
      return false;
    unsigned dimension = 0;
    for (int32_t size : ll.getOutDimSizes())
      if (int64_t(size) != ty.getShape()[dimension++])
        return false;
    // Low offset bits span the whole row/column plane without touching batch.
    // High bits permute only batch. Invertibility proves both maps bijective;
    // no particular logical row order or batch-bit permutation is assumed.
    auto bases = ll.getBases().lookup(offset);
    unsigned lowBits = llvm::Log2_64(planeElements);
    if (bases.size() != llvm::Log2_64(elements))
      return false;
    for (auto [i, basis] : llvm::enumerate(bases)) {
      if (basis.size() != 3)
        return false;
      if (i < lowBits) {
        if (basis[0] != 0)
          return false;
      } else if (basis[1] != 0 || basis[2] != 0) {
        return false;
      }
    }
    if (enc.getIntervals().size() != enc.getPaddings().size())
      return false;
    for (auto [interval, padding] :
         llvm::zip_equal(enc.getIntervals(), enc.getPaddings()))
      if (!llvm::isPowerOf2_32(interval) ||
          !llvm::isPowerOf2_32(padding) || planeElements % interval)
        return false;
    auto physical = [&](uint64_t n) -> std::optional<uint64_t> {
      uint64_t result = n;
      for (auto [interval, padding] :
           llvm::zip_equal(enc.getIntervals(), enc.getPaddings())) {
        uint64_t extra;
        if (!ttg::paddedSlotMul(n / interval, padding, extra) ||
            !ttg::paddedSlotAdd(result, extra, result))
          return std::nullopt;
      }
      return result;
    };
    auto stride = physical(planeElements);
    auto last = physical(planeElements - 1);
    auto rootLast = physical(elements - 1);
    uint64_t footprint, rootElements, rootBytes;
    if (!stride || !last || !rootLast ||
        !ttg::paddedSlotAdd(*last, 1, footprint) || footprint > *stride ||
        !ttg::paddedSlotAdd(*rootLast, 1, rootElements) ||
        !ttg::paddedSlotMul(rootElements, 2, rootBytes) ||
        rootBytes > uint64_t(std::numeric_limits<int32_t>::max()) ||
        rootBytes > allocation->getAllocatedSize(id))
      return false;
    return true;
  }

  // A bounded same-rank subview never changes the physical batch plane:
  // the allocation/encoding stay identical and every logical coordinate is
  // contained in its source. In particular, batch offsets and slicing decline.
  bool boundedPaddedBatchSlice(
      mlir::triton::gpu::MemDescSubsliceOp slice) const {
    namespace ttg = mlir::triton::gpu;
    auto src = slice.getSrc().getType(), dst = slice.getType();
    auto offsets = slice.getOffsets();
    if (src.getRank() != 3 || dst.getRank() != 3 || offsets.size() != 3 ||
        !mlir::isa<ttg::PaddedSharedEncodingAttr>(src.getEncoding()) ||
        src.getEncoding() != dst.getEncoding() ||
        src.getAllocShape() != dst.getAllocShape() ||
        src.getElementType() != dst.getElementType() ||
        src.getMemorySpace() != dst.getMemorySpace() ||
        src.getMutableMemory() != dst.getMutableMemory() ||
        offsets[0] != 0 || src.getShape()[0] != dst.getShape()[0])
      return false;
    for (unsigned d = 0; d != 3; ++d) {
      int64_t begin = offsets[d], extent = dst.getShape()[d];
      // Subtraction form also excludes offset+extent signed overflow.
      if (begin < 0 || extent <= 0 || src.getShape()[d] < extent ||
          begin > src.getShape()[d] - extent)
        return false;
    }
    return true;
  }

  // Transpose changes logical axes only, including the allocation shape and
  // exact full-allocation linear map. A retained axis0 cannot change ownership.
  bool paddedBatchViewTranspose(
      mlir::triton::gpu::MemDescTransOp trans) const {
    namespace ttg = mlir::triton::gpu;
    auto src = trans.getSrc().getType(), dst = trans.getType();
    auto a = mlir::dyn_cast<ttg::PaddedSharedEncodingAttr>(src.getEncoding());
    auto b = mlir::dyn_cast<ttg::PaddedSharedEncodingAttr>(dst.getEncoding());
    auto order = trans.getOrder();
    if (!a || !b || src.getRank() != 3 || dst.getRank() != 3 ||
        order.size() != 3 || order[0] != 0 ||
        src.getAllocShape().size() != 3 || dst.getAllocShape().size() != 3 ||
        src.getElementType() != dst.getElementType() ||
        src.getMemorySpace() != dst.getMemorySpace() ||
        src.getMutableMemory() != dst.getMutableMemory() ||
        a.getIntervals() != b.getIntervals() || a.getPaddings() != b.getPaddings())
      return false;
    llvm::SmallVector<bool, 3> seen(3, false);
    llvm::SmallVector<int, 3> permutation;
    for (unsigned d = 0; d != 3; ++d) {
      int axis = order[d];
      if (axis < 0 || axis >= 3 || seen[axis] ||
          dst.getShape()[d] != src.getShape()[axis] ||
          dst.getAllocShape()[d] != src.getAllocShape()[axis])
        return false;
      seen[axis] = true;
      permutation.push_back(axis);
    }
    return mlir::triton::transposeLinearLayout(a.getLinearComponent(), permutation) ==
           b.getLinearComponent();
  }

  // A normalized view must lead to a unique explicit allocation. For Padded,
  // direct rank3 roots, bounded batch-preserving slices and exact transposes
  // are accepted. Reshapes and indexed Padded stage roots decline per ID.
  // The legacy Swizzled leading-stage path is retained only for a bounded
  // CONSTANT full-plane index; dynamic or phi-dependent indices decline.
  std::optional<Provenance> provenance(mlir::Value value, Id id) const {
    using namespace mlir;
    namespace ttg = triton::gpu;
    value = identicalIncoming(value);
    for (unsigned depth = 0; value && depth < 8; ++depth) {
      if (auto trans = value.getDefiningOp<ttg::MemDescTransOp>()) {
        if (!paddedBatchViewTranspose(trans))
          return std::nullopt;
        value = identicalIncoming(trans.getSrc());
        continue;
      }
      if (auto slice = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
        if (!boundedPaddedBatchSlice(slice))
          return std::nullopt;
        value = identicalIncoming(slice.getSrc());
        continue;
      }
      if (auto alloc = value.getDefiningOp<ttg::LocalAllocOp>()) {
        if (alloc.isSharedMemoryAlloc() &&
            allocation->getBufferId(value) == id && roots.lookup(id) == value)
          return Provenance{value, alloc.getType()};
        return std::nullopt;
      }
      if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>()) {
        auto src = index.getSrc().getType(), dst = index.getType();
        auto root = identicalIncoming(index.getSrc());
        auto alloc = root ? root.getDefiningOp<ttg::LocalAllocOp>()
                          : ttg::LocalAllocOp{};
        llvm::APInt slot;
        if (!alloc || !alloc.isSharedMemoryAlloc() ||
            !isa<ttg::SwizzledSharedEncodingAttr>(dst.getEncoding()) ||
            src.getRank() != 4 || dst.getRank() != 3 ||
            src.getShape().drop_front() != dst.getShape() ||
            src.getEncoding() != dst.getEncoding() ||
            src.getElementType() != dst.getElementType() ||
            src.getMemorySpace() != dst.getMemorySpace() ||
            !matchPattern(index.getIndex(), m_ConstantInt(&slot)) ||
            slot.getBitWidth() > 32 || slot.isNegative() ||
            slot.getZExtValue() >= uint64_t(src.getShape()[0]) ||
            allocation->getBufferId(root) != id || roots.lookup(id) != root)
          return std::nullopt;
        return Provenance{root, dst};
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  bool validPlane(MemDesc ty, Id id) const {
    using namespace mlir;
    namespace ttg = triton::gpu;
    if (ty.getRank() != 3 ||
        !(ty.getElementType().isF16() || ty.getElementType().isBF16()) ||
        !isa<ttg::SharedMemorySpaceAttr>(ty.getMemorySpace()) ||
        !llvm::all_of(ty.getShape(), [](int64_t size) {
          return size > 0 && llvm::isPowerOf2_64(size);
        }))
      return false;
    if (isa<ttg::PaddedSharedEncodingAttr>(ty.getEncoding()))
      return paddedPlanes(ty, id);
    auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(ty.getEncoding());
    return shared && shared.getOrder().size() == 3 &&
           shared.getOrder().back() == 0;
  }

  // A recycled allocation can also be wave-private, but equal byte offsets
  // alone do not prove it. Derive the physical batch-plane pitch from the
  // actual swizzled chart, and require every overlapping byte to have the same
  // wave owner in both allocations. Unproved C scratch, views and padded charts
  // retain the existing allocation-ID rule.
  std::optional<std::pair<uint64_t, uint64_t>> wavePartition(Id id) const {
    using namespace mlir;
    namespace ttg = triton::gpu;
    auto root = roots.lookup(id);
    auto type = root ? dyn_cast<MemDesc>(root.getType()) : MemDesc{};
    if (!type || type != canonicalTypes.lookup(id) || type.getRank() != 3 ||
        type.getShape() != type.getAllocShape() ||
        ArrayRef<int64_t>(ttg::getAllocationShapePerCTA(type)) != type.getShape() ||
        !isa<ttg::SwizzledSharedEncodingAttr>(type.getEncoding()))
      return std::nullopt;
    uint64_t plane, elements, bytes;
    if (!ttg::paddedSlotMul(uint64_t(type.getDimSize(1)),
                            uint64_t(type.getDimSize(2)), plane) ||
        !ttg::paddedSlotMul(plane, uint64_t(type.getDimSize(0)), elements) ||
        elements > uint64_t(std::numeric_limits<int32_t>::max()) ||
        !ttg::paddedSlotMul(elements, 2, bytes) ||
        bytes > allocation->getAllocatedSize(id))
      return std::nullopt;
    auto layout = ttg::toLinearLayout(type);
    auto offset = StringAttr::get(type.getContext(), "offset");
    auto block = StringAttr::get(type.getContext(), "block");
    if (!layout.isInvertible() || layout.getInDimSize(block) != 1 ||
        uint64_t(layout.getInDimSize(offset)) != elements)
      return std::nullopt;
    auto bases = layout.getBases().lookup(offset);
    unsigned lowBits = llvm::Log2_64(plane);
    if (bases.size() != llvm::Log2_64(elements))
      return std::nullopt;
    for (auto [bit, basis] : llvm::enumerate(bases)) {
      if (basis.size() != 3 ||
          (bit < lowBits ? basis[0] != 0
                         : basis != std::vector<int32_t>(
                               {int32_t(1u << (bit - lowBits)), 0, 0})))
        return std::nullopt;
    }
    uint64_t planeBytes;
    if (!ttg::paddedSlotMul(plane, 2, planeBytes))
      return std::nullopt;
    return std::make_pair(uint64_t(allocation->getOffset(id)), planeBytes);
  }

  bool sameWavePartition(Id a, Id b) const {
    auto left = wavePartition(a), right = wavePartition(b);
    if (!left || !right || left->second != right->second)
      return false;
    auto warps = mod->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps");
    uint64_t period;
    if (!warps || warps.getInt() <= 0 ||
        !mlir::triton::gpu::paddedSlotMul(
            left->second, uint64_t(warps.getInt()), period) || !period)
      return false;
    return left->first % period == right->first % period;
  }

public:
  WavePrivateSyncStaging(mlir::ModuleOp mod, mlir::ModuleAllocation &moduleAlloc,
                        const mlir::triton::HCU::TargetInfo &target)
      : mod(mod), target(target) {
    using namespace mlir;
    namespace ttg = triton::gpu;
    namespace hcu = triton::hcugpu;
    auto opt = mod->getAttrOfType<IntegerAttr>("ttg.hcu.wave_private_sync");
    if (!opt || opt.getInt() != 1)
      return;
    if (target.getArch() != "gfx936" ||
        ttg::TritonGPUDialect::getNumCTAs(mod) != 1 ||
        ttg::TritonGPUDialect::getThreadsPerWarp(mod) != 64)
      return;
    SmallVector<FunctionOpInterface> funcs;
    mod.walk([&](Operation *op) {
      if (auto fn = dyn_cast<FunctionOpInterface>(op))
        funcs.push_back(fn);
    });
    if (funcs.size() != 1)
      return;
    allocation = moduleAlloc.getFuncData(funcs.front());
    bool unsupported = false;
    // Calls, opaque inline asm and asynchronous LDS provenance are never
    // contained by the per-allocation view rejection below.
    mod.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (isa<CallOpInterface, triton::ExternElementwiseOp,
              ttg::AsyncCopyGlobalToLocalOp,
              hcu::BufferLoadToLocalOp, hcu::MatrixLoadToLocalOp>(op) ||
          name.contains("async") || name.contains("tdm") ||
          name.contains("asm") || name.contains("call"))
        unsupported = true;
      if (auto load = dyn_cast<ttg::LocalLoadOp>(op))
        if (load.getToken())
          unsupported = true;
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (!alloc.isSharedMemoryAlloc())
          return;
        Id id = allocation->getBufferId(alloc.getResult());
        if (id == Allocation::InvalidBufferId || roots.count(id))
          unsupported = true;
        else
          roots[id] = alloc.getResult();
      }
    });
    if (unsupported)
      return;
    // Every descriptor must have known explicit roots. A known unsupported C
    // view rejects its complete allocation, not unrelated proven A/B roots.
    mod.walk([&](Operation *op) {
      SmallVector<Value> descs;
      for (Value value : op->getOperands())
        if (isa<MemDesc>(value.getType()))
          descs.push_back(value);
      for (Value value : op->getResults())
        if (isa<MemDesc>(value.getType()))
          descs.push_back(value);
      for (Value value : descs) {
        auto ids = allocation->getBufferIds(value);
        if (ids.empty())
          unsupported = true;
        for (Id id : ids)
          if (!roots.count(id))
            unsupported = true;
      }
      if (descs.empty())
        return;
      bool allowed = isa<ttg::LocalAllocOp, ttg::LocalDeallocOp,
                         ttg::LocalLoadOp, ttg::LocalStoreOp,
                         BranchOpInterface>(op);
      if (auto trans = dyn_cast<ttg::MemDescTransOp>(op))
        allowed = paddedBatchViewTranspose(trans);
      if (auto slice = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
        auto ids = allocation->getBufferIds(slice.getResult());
        allowed = ids.size() == 1 &&
                  provenance(slice.getResult(), *ids.begin()).has_value();
      }
      if (auto index = dyn_cast<ttg::MemDescIndexOp>(op)) {
        auto ids = allocation->getBufferIds(index.getResult());
        allowed = ids.size() == 1 &&
                  provenance(index.getResult(), *ids.begin()).has_value();
      }
      if (!allowed)
        for (Value value : descs)
          for (Id id : allocation->getBufferIds(value))
            rejected.insert(id);
    });
    // Inventory exact ordinary accesses. All invalid accesses reject their
    // entire known ID, including a different alias used elsewhere in the CFG.
    mod.walk([&](Operation *op) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      if (!effects)
        return;
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto effect : instances) {
        auto value = effect.getValue();
        if (!value || !isa<MemDesc>(value.getType()) ||
            !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()))
          continue;
        auto ids = allocation->getBufferIds(value);
        auto view = ids.size() == 1
                        ? provenance(value, *ids.begin())
                        : std::nullopt;
        bool valid = view && view->root == roots.lookup(*ids.begin()) &&
                     validPlane(view->plane, *ids.begin());
        if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
          auto source = dyn_cast<RankedTensorType>(store.getSrc().getType());
          valid = valid && source && store.getDst() == value &&
                  isa<MemoryEffects::Write>(effect.getEffect()) &&
                  source.getShape() == cast<MemDesc>(value.getType()).getShape() &&
                  ownedByBatchWave(source);
        } else if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
          auto source = alloc.getSrc()
                            ? dyn_cast<RankedTensorType>(alloc.getSrc().getType())
                            : RankedTensorType{};
          valid = valid && source && alloc.getResult() == value &&
                  isa<MemoryEffects::Write>(effect.getEffect()) &&
                  source.getShape() == cast<MemDesc>(value.getType()).getShape() &&
                  ownedByBatchWave(source);
        } else if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
          auto result = dyn_cast<RankedTensorType>(load.getType());
          auto dot = result ? dyn_cast<ttg::DotOperandEncodingAttr>(result.getEncoding())
                            : ttg::DotOperandEncodingAttr{};
          valid = valid && !load.getToken() && load.getSrc() == value &&
                  result && dot &&
                  isa<ttg::AMDMfmaEncodingAttr>(dot.getParent()) &&
                  isa<MemoryEffects::Read>(effect.getEffect()) &&
                  result.getShape() == cast<MemDesc>(value.getType()).getShape() &&
                  ownedByBatchWave(result);
        } else {
          valid = false;
        }
        for (Id id : ids) {
          if (!valid ||
              (canonicalTypes.count(id) &&
               canonicalTypes.lookup(id) != view->plane)) {
            rejected.insert(id);
            continue;
          }
          canonicalTypes[id] = view->plane;
          if (isa<ttg::PaddedSharedEncodingAttr>(view->plane.getEncoding()))
            paddedIds.insert(id);
          accesses[op] = id;
        }
      }
    });
    for (auto it = accesses.begin(); it != accesses.end();) {
      if (rejected.contains(it->second))
        accesses.erase(it++);
      else
        ++it;
    }
    bool writer = llvm::any_of(accesses, [](const auto &entry) {
      return isa<ttg::LocalStoreOp, ttg::LocalAllocOp>(entry.first);
    });
    if (!writer || unsupported)
      accesses.clear();
  }

  // A recycled C allocation remains unproved even at the same LDS address.
  // Different proven input IDs may share storage only with identical physical
  // wave partitions; shape/offset equality by itself grants nothing.
  bool filter(mlir::Operation *before, mlir::Operation *after) const {
    auto a = accesses.find(before), b = accesses.find(after);
    if (a != accesses.end() || b != accesses.end())
      return a != accesses.end() && b != accesses.end() &&
             (a->second == b->second || sameWavePartition(a->second, b->second));
    return mlir::triton::HCU::membarFilter(before, after);
  }

  void prepare() {
    using namespace mlir;
    if (accesses.empty())
      return;
    // Decide on the original IR, in real block order, before inserting any
    // waits/fences. DenseMap iteration order is not program order. Every block
    // starts a fresh run: no predecessor, region, loop or merge fact is reused.
    auto readRunPure = [](Operation *op) {
      if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
          isa<CallOpInterface>(op) || !isMemoryEffectFree(op))
        return false;
      auto name = op->getName().getStringRef();
      // MemoryEffectFree alone is insufficient: e.g. ttg.convert_layout is
      // Pure in TTGIR but can lower to LDS scratch accesses. Only explicit
      // arithmetic and register/view operations can bridge this read run.
      if (name.starts_with("arith."))
        return true;
      return name == "tt.make_range" || name == "tt.splat" ||
             name == "tt.broadcast" || name == "tt.expand_dims" ||
             name == "tt.addptr" || name == "tt.get_program_id" ||
             name == "tt.get_num_programs" ||
             name == "ttg.memdesc_index" || name == "ttg.memdesc_trans" ||
             name == "ttg.memdesc_subslice" || name == "ttg.memdesc_reshape" ||
             name == "ttg.memdesc_reinterpret";
    };
    // A tensor transpose lowers to the identical LLVM aggregate (the verifier
    // propagates the permuted encoding). It may bridge writers without LDS
    // traffic. Do not admit convert_layout: Pure TTGIR can need LDS scratch.
    auto writerRunPure = [&](Operation *op) {
      if (isa<triton::TransOp>(op))
        return op->getNumRegions() == 0 && op->getNumSuccessors() == 0 &&
               isMemoryEffectFree(op);
      return readRunPure(op);
    };
    using ByteRange = std::pair<size_t, size_t>; // [begin, end), full root.
    auto allocatedBytes = [&](Id id) -> std::optional<ByteRange> {
      if (id == Allocation::InvalidBufferId || !roots.count(id))
        return std::nullopt;
      size_t begin = allocation->getOffset(id);
      size_t size = allocation->getAllocatedSize(id);
      if (!size || size > std::numeric_limits<size_t>::max() - begin)
        return std::nullopt;
      return ByteRange{begin, begin + size};
    };
    llvm::DenseSet<Operation *> waitBefore;
    mod.walk([&](Operation *owner) {
      for (Region &region : owner->getRegions()) {
        for (Block &block : region) {
          bool readRun = false;
          SmallVector<Id, 4> writerIds;
          SmallVector<ByteRange, 4> writerBytes;
          auto resetWriters = [&]() {
            writerIds.clear();
            writerBytes.clear();
          };
          for (Operation &op : block) {
            auto access = accesses.find(&op);
            if (access != accesses.end()) {
              if (isa<triton::gpu::LocalLoadOp>(&op)) {
                // First read drains ALL prior writes. Preserve read-run rules.
                if (!readRun)
                  waitBefore.insert(&op);
                readRun = true;
                resetWriters();
              } else {
                auto current = allocatedBytes(access->second);
                bool disjointRun = current && !writerIds.empty() &&
                    !llvm::is_contained(writerIds, access->second) &&
                    llvm::all_of(writerBytes, [&](const ByteRange &previous) {
                      return current->second <= previous.first ||
                             previous.second <= current->first;
                    });
                // First writer drains preceding reads/writes. Only subsequent
                // writes to pairwise disjoint FULL roots may share that drain.
                // Same ID (even different slots), recycled/overlapping roots,
                // zero sizes and overflow restart conservatively with a wait.
                if (!disjointRun) {
                  waitBefore.insert(&op);
                  resetWriters();
                }
                if (current) {
                  writerIds.push_back(access->second);
                  writerBytes.push_back(*current);
                }
                readRun = false;
              }
            } else {
              if (!readRunPure(&op))
                readRun = false;
              if (!writerRunPure(&op))
                resetWriters();
            }
          }
        }
      }
    });
    IRRewriter builder(mod->getContext());
    unsigned paddedAccesses = 0;
    for (auto [op, id] : accesses) {
      builder.setInsertionPoint(op);
      // Full waits remain before each first writer/read and every unproved
      // writer transition. Every access retains the original wave fence.
      if (waitBefore.contains(op))
        triton::hcugpu::MemoryCounterWaitOp::create(
            builder, op->getLoc(), nullptr, nullptr, builder.getI32IntegerAttr(0));
      target.barrier(op->getLoc(), builder, /*isWarpSync=*/true);
      paddedAccesses += paddedIds.contains(id);
    }
    mod->setAttr("ttg.hcu.wave_private_sync_accesses",
                 builder.getI32IntegerAttr(accesses.size()));
    mod->setAttr("ttg.hcu.wave_private_sync_padded_accesses",
                 builder.getI32IntegerAttr(paddedAccesses));
  }
};
} // namespace
