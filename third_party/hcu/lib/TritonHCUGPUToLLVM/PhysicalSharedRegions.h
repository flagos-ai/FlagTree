#ifndef TRITON_HCU_PHYSICAL_SHARED_REGIONS_H
#define TRITON_HCU_PHYSICAL_SHARED_REGIONS_H

#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "WarpSpecializeUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <optional>

namespace mlir::triton::HCU {

// Read-only, invocation-local physical byte footprints for ordinary shared
// descriptors. No frontend witness, packet role, model shape, or pipeline
// template grants disjointness. Unknown views/edges and unanchored cycles fail
// closed; dynamic indices denote unions, never presumed distinct iterations.
class PhysicalSharedRegions {
  using MD = triton::gpu::MemDescType;
  struct View {
    triton::gpu::LocalAllocOp root;
    uint64_t base = 0; // bytes relative to the actual allocation
    SmallVector<int32_t> offsets;
  };
  struct Region {
    Operation *root = nullptr;
    llvm::BitVector bytes;
  };
  ModuleAllocation &allocation;
  llvm::DenseMap<Operation *, Region> accesses;
  static constexpr uint64_t PointBudget = uint64_t(1) << 20;

  static std::optional<unsigned> elementBytes(MD type) {
    auto elem = type.getElementType();
    if (!elem.isIntOrFloat()) return std::nullopt;
    unsigned bits = elem.getIntOrFloatBitWidth();
    if (!bits || bits % 8) return std::nullopt;
    return bits / 8;
  }
  static bool ordinary(MD type) {
    if (!type || !elementBytes(type) ||
        !isa<triton::gpu::SharedLinearEncodingAttr,
             triton::gpu::SwizzledSharedEncodingAttr>(type.getEncoding()))
      return false;
    return llvm::all_of(type.getAllocShape(), [](int64_t n) {
      return n > 0 && llvm::isPowerOf2_64(uint64_t(n));
    });
  }
  static std::optional<uint64_t> constant(Value value) {
    auto c = value.getDefiningOp<arith::ConstantOp>();
    auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
    if (!a || a.getValue().getBitWidth() > 64 || a.getValue().isNegative())
      return std::nullopt;
    return a.getValue().getZExtValue();
  }
  static std::optional<uint64_t> logicalElements(MD type) {
    uint64_t n = 1;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || uint64_t(dim) > PointBudget / n) return std::nullopt;
      n *= dim;
    }
    return n;
  }
  // Check the supported chart domain before invoking conversion helpers that
  // assert on rank/extent mismatch (e.g. a whole multibuffer reinterpret).
  // Unmodeled legal views retain the generic memory-dependency analysis.
  static std::optional<LinearLayout> chart(MD type) {
    if (auto shared = dyn_cast<triton::gpu::SharedLinearEncodingAttr>(type.getEncoding())) {
      auto ll = shared.getLinearLayout();
      auto shape = type.getAllocShape().take_back(type.getRank());
      if (ll.getNumOutDims() != shape.size() ||
          !llvm::equal(ll.getOutDimSizes(), shape)) return std::nullopt;
      return ll;
    }
    auto shared = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(type.getEncoding());
    if (!shared || shared.getOrder().size() != size_t(type.getRank()))
      return std::nullopt;
    return triton::gpu::toLinearLayout(type);
  }
  // Mirror SharedMemoryObject::getShmemOffset, with compile-time offsets.
  // A bijection avoids claiming a unique address for broadcast storage.
  static std::optional<uint64_t> affine(MD type, const View &view) {
    if (view.offsets.size() != size_t(type.getRank())) return std::nullopt;
    auto maybe = chart(type);
    if (!maybe) return std::nullopt;
    if (!LLVM::SharedMemoryObject::isAffineSharedMemoryAccess(type)) return 0;
    auto ll = *maybe;
    auto offset = StringAttr::get(type.getContext(), "offset");
    auto dims = llvm::to_vector(ll.getOutDimNames());
    ll = ll.sublayout({offset}, dims);
    if (!ll.isSurjective() || ll.getTotalInDimSize() != ll.getTotalOutDimSize())
      return std::nullopt;
    SmallVector<std::pair<StringAttr, int32_t>> inputs;
    for (auto [i, dim] : llvm::enumerate(dims)) {
      if (view.offsets[i] < 0 || view.offsets[i] >= ll.getOutDimSize(dim))
        return std::nullopt;
      inputs.push_back({dim, view.offsets[i]});
    }
    return uint64_t(ll.invert().apply(inputs)[0].second);
  }
  std::optional<uint64_t> rootBytes(triton::gpu::LocalAllocOp root) const {
    auto type = root.getType();
    if (!ordinary(type) || !logicalElements(type) || !root.isSharedMemoryAlloc() ||
        type.getAllocShape() != type.getShape()) return std::nullopt;
    auto function = root->getParentOfType<triton::FuncOp>();
    auto *data = function ? allocation.getFuncData(function) : nullptr;
    if (!data) return std::nullopt;
    auto id = data->getBufferId(root.getResult());
    if (id == Allocation::InvalidBufferId) return std::nullopt;
    uint64_t size = data->getAllocatedSize(id);
    auto offset = root->getAttrOfType<IntegerAttr>("allocation.offset");
    uint64_t total = allocation.getSharedMemorySize();
    if (!size || size > PointBudget || !offset || offset.getInt() < 0 ||
        uint64_t(offset.getInt()) != data->getOffset(id) ||
        data->getOffset(id) > total || size > total - data->getOffset(id))
      return std::nullopt;
    return size;
  }
  // Evaluate the exact active logical domain through the same inverse shared
  // chart and XOR affine offset used by ordinary LocalLoad/LocalStore lowering.
  std::optional<llvm::BitVector> footprint(MD type, const View &view,
                                         uint64_t &budget) const {
    auto count = logicalElements(type), bytes = rootBytes(view.root);
    auto elem = elementBytes(type);
    auto origin = affine(type, view);
    if (!count || !bytes || !elem || !origin || *count > budget)
      return std::nullopt;
    budget -= *count;
    auto maybe = chart(type);
    if (!maybe) return std::nullopt;
    auto ll = *maybe;
    auto off = StringAttr::get(type.getContext(), "offset");
    auto dims = llvm::to_vector(ll.getOutDimNames());
    ll = ll.sublayout({off}, dims);
    if (dims.size() != size_t(type.getRank()) || !ll.isSurjective() ||
        ll.getTotalInDimSize() != ll.getTotalOutDimSize()) return std::nullopt;
    auto inverse = ll.invert();
    llvm::BitVector result(*bytes, false);
    for (uint64_t index = 0; index < *count; ++index) {
      uint64_t remainder = index;
      SmallVector<std::pair<StringAttr, int32_t>> xy(dims.size());
      for (int d = type.getRank() - 1; d >= 0; --d) {
        xy[d] = {dims[d], int32_t(remainder % type.getShape()[d])};
        remainder /= type.getShape()[d];
      }
      uint64_t relative = uint32_t(inverse.apply(xy)[0].second) ^ *origin;
      if (relative > (*bytes - std::min<uint64_t>(*bytes, *elem)) / *elem)
        return std::nullopt;
      uint64_t address = view.base + relative * *elem;
      if (address > *bytes || *elem > *bytes - address) return std::nullopt;
      result.set(address, address + *elem);
    }
    return result;
  }
  // Resolve straight view chains. Control-flow forwarding is handled separately
  // below, so recursion cannot silently accept a cycle as an allocation root.
  std::optional<SmallVector<View>> views(Value value, unsigned depth,
                                        uint64_t &budget) const {
    value = resolveWarpSpecializeCapture(value);
    auto type = dyn_cast<MD>(value.getType());
    if (depth > 64 || !ordinary(type)) return std::nullopt;
    if (auto root = value.getDefiningOp<triton::gpu::LocalAllocOp>()) {
      if (!rootBytes(root)) return std::nullopt;
      return SmallVector<View>{View{root, 0, SmallVector<int32_t>(type.getRank(), 0)}};
    }
    if (auto index = value.getDefiningOp<triton::gpu::MemDescIndexOp>()) {
      auto input = views(index.getSrc(), depth + 1, budget);
      auto src = index.getSrc().getType();
      if (!input || src.getRank() != type.getRank() + 1 ||
          src.getElementType() != type.getElementType()) return std::nullopt;
      uint64_t bound;
      bool singleton = false;
      if (auto c = constant(index.getIndex())) { bound = *c; singleton = true; }
      else if (auto mask = index.getIndex().getDefiningOp<arith::AndIOp>()) {
        auto c = constant(mask.getLhs());
        if (!c) c = constant(mask.getRhs());
        if (!c) return std::nullopt;
        bound = *c;
      } else return std::nullopt;
      if (bound >= uint64_t(src.getShape()[0]) ||
          bound >= PointBudget || input->size() > PointBudget / (bound + 1))
        return std::nullopt;
      uint64_t stride = product(triton::gpu::getAllocationShapePerCTA(
          type.getEncoding(), type.getAllocShape().take_back(type.getRank())));
      SmallVector<View> output;
      for (auto view : *input) {
        if (view.offsets.size() != size_t(src.getRank())) return std::nullopt;
        int32_t leading = view.offsets.front();
        view.offsets.erase(view.offsets.begin());
        for (uint64_t i = singleton ? bound : 0; i <= bound; ++i) {
          View next = view;
          uint64_t effective = i + leading;
          if (leading < 0 || effective >= uint64_t(src.getAllocShape().take_back(src.getRank())[0]) ||
              stride > PointBudget / (effective + 1)) return std::nullopt;
          next.base += effective * stride * *elementBytes(type);
          output.push_back(std::move(next));
        }
      }
      return output;
    }
    if (auto slice = value.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
      auto input = views(slice.getSrc(), depth + 1, budget);
      auto src = slice.getSrc().getType();
      if (!input || src.getRank() != type.getRank() ||
          src.getEncoding() != type.getEncoding() ||
          src.getElementType() != type.getElementType() ||
          src.getAllocShape() != type.getAllocShape()) return std::nullopt;
      for (auto &view : *input) {
        for (unsigned d = 0; d < view.offsets.size(); ++d) {
          int64_t offset = slice.getOffsets()[d];
          if (offset < 0 || offset > src.getShape()[d] ||
              type.getShape()[d] > src.getShape()[d] - offset ||
              offset > INT32_MAX - view.offsets[d]) return std::nullopt;
          view.offsets[d] += offset;
        }
      }
      return input;
    }
    if (auto cast = value.getDefiningOp<triton::gpu::MemDescReinterpretOp>()) {
      auto input = views(cast.getSrc(), depth + 1, budget);
      auto src = cast.getSrc().getType();
      if (!input || src.getMemorySpace() != type.getMemorySpace() ||
          type.getShape() != type.getAllocShape()) return std::nullopt;
      auto destCount = logicalElements(type);
      if (!destCount) return std::nullopt;
      for (auto &view : *input) {
        auto region = footprint(src, view, budget);
        auto origin = affine(src, view);
        if (!region || !origin || region->none()) return std::nullopt;
        uint64_t base = view.base + *origin * *elementBytes(src);
        uint64_t size = region->count();
        // The reinterpret contract requires contiguous source storage. Prove
        // both its origin and complete interval; no shape-name inference.
        if (uint64_t(region->find_first()) != base ||
            uint64_t(region->find_last()) + 1 != base + size ||
            *destCount > size / *elementBytes(type)) return std::nullopt;
        view.base = base;
        view.offsets.assign(type.getRank(), 0);
      }
      return input;
    }
    return std::nullopt;
  }
  std::optional<Region> resolve(Value source) const {
    llvm::DenseSet<Value> seen, anchored;
    llvm::DenseMap<Value, SmallVector<Value>> dependents;
    SmallVector<Value> pending{source}, seeds;
    uint64_t budget = PointBudget;
    Region result;
    while (!pending.empty()) {
      Value v = pending.pop_back_val();
      if (!seen.insert(v).second) continue;
      if (seen.size() > 4096) return std::nullopt;
      auto follow = [&](Value input) {
        dependents[input].push_back(v); pending.push_back(input);
      };
      if (Value capture = resolveWarpSpecializeCapture(v); capture != v) {
        follow(capture);
      } else if (auto select = v.getDefiningOp<arith::SelectOp>()) {
        follow(select.getTrueValue()); follow(select.getFalseValue());
      } else if (auto arg = dyn_cast<BlockArgument>(v)) {
        Block *block = arg.getOwner();
        bool incoming = false;
        for (Block *pred : block->getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch || !isa<cf::BranchOp, cf::CondBranchOp>(pred->getTerminator()))
            return std::nullopt;
          for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
            if (branch->getSuccessor(i) != block) continue;
            auto ops = branch.getSuccessorOperands(i);
            unsigned produced = ops.getProducedOperandCount(), n = arg.getArgNumber();
            if (n < produced || n - produced >= ops.getForwardedOperands().size())
              return std::nullopt;
            follow(ops.getForwardedOperands()[n - produced]); incoming = true;
          }
        }
        if (!incoming) return std::nullopt;
      } else {
        auto states = views(v, 0, budget);
        if (!states) return std::nullopt;
        auto type = cast<MD>(v.getType());
        for (auto view : *states) {
          auto bytes = footprint(type, view, budget);
          if (!bytes || (result.root && result.root != view.root.getOperation()))
            return std::nullopt;
          if (!result.root) { result.root = view.root; result.bytes = *bytes; }
          else result.bytes |= *bytes;
        }
        seeds.push_back(v);
      }
    }
    while (!seeds.empty()) {
      Value v = seeds.pop_back_val();
      if (!anchored.insert(v).second) continue;
      for (Value dependent : dependents[v]) seeds.push_back(dependent);
    }
    if (!result.root || result.bytes.none() || anchored.size() != seen.size())
      return std::nullopt;
    return result;
  }
public:
  PhysicalSharedRegions(ModuleOp mod, ModuleAllocation &allocation)
      : allocation(allocation) {
    if (triton::gpu::TritonGPUDialect::getNumCTAs(mod) != 1) return;
    mod.walk([&](Operation *op) {
      Value value;
      if (auto load = dyn_cast<triton::gpu::LocalLoadOp>(op)) value = load.getSrc();
      else if (auto store = dyn_cast<triton::gpu::LocalStoreOp>(op)) value = store.getDst();
      else if (auto copy = dyn_cast<triton::hcugpu::BufferLoadToLocalOp>(op)) value = copy.getDest();
      else if (auto copy = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op)) value = copy.getDest();
      else return;
      if (auto region = resolve(value)) accesses[op] = std::move(*region);
    });
  }
  // Producer eligibility needs each possible view to be a contiguous actual
  // byte interval, rather than treating the union of ring slots as one view.
  bool contiguousAligned(Value source, unsigned alignment) const {
    auto type = dyn_cast<MD>(source.getType());
    if (!ordinary(type) || !alignment) return false;
    uint64_t budget = PointBudget;
    auto states = views(source, 0, budget);
    auto count = logicalElements(type);
    if (!states || !count || states->empty()) return false;
    for (auto view : *states) {
      auto bytes = footprint(type, view, budget);
      auto rootOffset = view.root->getAttrOfType<IntegerAttr>("allocation.offset");
      if (!bytes || bytes->none() || !rootOffset ||
          view.root.getAlignmentOrDefault() < alignment ||
          view.root.getAlignmentOrDefault() % alignment ||
          (uint64_t(bytes->find_first()) + rootOffset.getInt()) % alignment ||
          bytes->count() != *count * *elementBytes(type) ||
          uint64_t(bytes->find_last() - bytes->find_first()) + 1 != bytes->count())
        return false;
    }
    return true;
  }
  bool disjoint(Operation *before, Operation *after) const {
    auto a = accesses.find(before), b = accesses.find(after);
    return a != accesses.end() && b != accesses.end() &&
        a->second.root == b->second.root &&
        !a->second.bytes.anyCommon(b->second.bytes);
  }
};
} // namespace mlir::triton::HCU
#endif
