#ifndef TRITON_HCU_WHOLE_PACKET_REGIONS_H
#define TRITON_HCU_WHOLE_PACKET_REGIONS_H

#include "Dialect/TritonHCUGPU/IR/WholePacketLayout.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::HCU {

// An overapproximation in PHYSICAL bytes. In particular, the opaque packet
// encoding is never treated as a logical LinearLayout. Every incoming CF edge
// must reach a checked flat owner/index/half/reinterpret chain; an unsupported
// edge or an unanchored cycle invalidates the entire queried value.
class WholePacketRegions {
  using MD = triton::gpu::MemDescType;
  using Packet = triton::gpu::HCUMlsSharedEncodingAttr;
  using Alloc = triton::gpu::LocalAllocOp;
  using Index = triton::gpu::MemDescIndexOp;
  using Slice = triton::gpu::MemDescSubsliceOp;
  using Reinterpret = triton::gpu::MemDescReinterpretOp;
  struct Region {
    Operation *owner = nullptr;
    uint32_t halves = 0;
    bool compact = false;
  };
  llvm::DenseMap<Operation *, Region> accesses;

  static std::optional<uint64_t> constant(Value v) {
    auto c = v.getDefiningOp<arith::ConstantOp>();
    auto i = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
    if (!i || i.getValue().getBitWidth() > 64 || i.getValue().isNegative())
      return std::nullopt;
    return i.getValue().getZExtValue();
  }

  static std::optional<Region> anchor(Value v, ModuleAllocation &allocation) {
    namespace wp = triton::hcugpu::wholepacket;
    auto md = dyn_cast<MD>(v.getType());
    auto enc = md ? dyn_cast<Packet>(md.getEncoding()) : Packet{};
    int64_t active = wp::Elements;
    if (enc && enc.getVersion() != 256) {
      if (md.getRank() != 2 || md.getShape()[0] < 1 || md.getShape()[0] > wp::Elements ||
          md.getShape()[1] < 1 || md.getShape()[1] > wp::Elements ||
          md.getShape()[0] * md.getShape()[1] > wp::Elements)
        return std::nullopt;
      active = md.getShape()[0] * md.getShape()[1];
    }
    auto view = v.getDefiningOp<Reinterpret>();
    auto slice = view ? view.getSrc().getDefiningOp<Slice>() : Slice{};
    auto index = slice ? slice.getSrc().getDefiningOp<Index>() : Index{};
    auto root = index ? index.getSrc().getDefiningOp<Alloc>() : Alloc{};
    bool compact=root && root.getType().getRank()==2 && root.getType().getShape()[1]==4096;
    const unsigned halfElements=compact?2048:wp::Elements;
    const unsigned frameBytes=compact?8192:wp::FrameBytes;
    auto compactEnabled=root?root->getParentOfType<ModuleOp>()->getAttrOfType<BoolAttr>("ttg.hcu.compact_packet_owner"):BoolAttr{};
    if(compact && (!compactEnabled || !compactEnabled.getValue() || !enc || enc.getVersion()==256 || active>2048))
      return std::nullopt;
    if (!enc || !wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind()) || enc.getOpIdx() > 1 ||
        enc.getElemBitWidth() != 16 || md.getRank() != 2 ||
        !llvm::equal(enc.getMlsTile(), md.getShape()) ||
        md.getElementType().getIntOrFloatBitWidth() != 16 ||
        !llvm::equal(enc.getCTALayout().getCTAsPerCGA(),
                     ArrayRef<unsigned>({1, 1})) ||
        !llvm::equal(enc.getCTALayout().getCTASplitNum(),
                     ArrayRef<unsigned>({1, 1})) ||
        !root || root.getSrc() || !root.isSharedMemoryAlloc() ||
        root.getType().getRank() != 2 ||
        root.getType().getAllocShape() != root.getType().getShape() ||
        root.getType().getShape()[1] != 2 * halfElements ||
        root.getType().getElementType() != md.getElementType() ||
        root.getAlignmentOrDefault() < frameBytes ||
        root.getAlignmentOrDefault() % frameBytes ||
        index.getType().getShape() != ArrayRef<int64_t>({int64_t(2 * halfElements)}) ||
        index.getType().getAllocShape() != index.getType().getShape() ||
        slice.getType().getShape() != ArrayRef<int64_t>({active}) ||
        slice.getType().getAllocShape() != index.getType().getShape() ||
        md.getAllocShape() != md.getShape() ||
        index.getType().getElementType() != md.getElementType() ||
        slice.getType().getElementType() != md.getElementType() ||
        index.getType().getMemorySpace() != root.getType().getMemorySpace() ||
        slice.getType().getMemorySpace() != root.getType().getMemorySpace() ||
        md.getMemorySpace() != root.getType().getMemorySpace() ||
        slice.getOffsets().size() != 1 ||
        slice.getOffsets()[0] != (enc.getOpIdx() == 0 ? halfElements : 0))
      return std::nullopt;
    auto flat = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
        root.getType().getEncoding());
    if (!flat || flat.getVec() != 1 || flat.getPerPhase() != 1 ||
        flat.getMaxPhase() != 1 || flat.getOrder() != ArrayRef<unsigned>({0}) ||
        index.getType().getEncoding() != flat ||
        slice.getType().getEncoding() != flat)
      return std::nullopt;
    int64_t count = root.getType().getShape()[0];
    if (count <= 0 || count > 65536 / frameBytes)
      return std::nullopt;
    auto function = root->getParentOfType<triton::FuncOp>();
    if (!function)
      return std::nullopt;
    auto *data = allocation.getFuncData(function);
    if (!data)
      return std::nullopt;
    auto id = data->getBufferId(root.getResult());
    auto offset = root->getAttrOfType<IntegerAttr>("allocation.offset");
    auto module = root->getParentOfType<ModuleOp>();
    auto emittedSize = module->getAttrOfType<IntegerAttr>("ttg.shared");
    uint64_t total = allocation.getSharedMemorySize();
    if (id == Allocation::InvalidBufferId || !offset || offset.getInt() < 0 ||
        !emittedSize || emittedSize.getInt() < 0 ||
        uint64_t(emittedSize.getInt()) != total || total > 65536 ||
        offset.getInt() != data->getOffset(id) ||
        data->getOffset(id) % frameBytes ||
        data->getOffset(id) > total ||
        uint64_t(count) * frameBytes > total - data->getOffset(id) ||
        data->getAllocatedSize(id) != uint64_t(count) * frameBytes)
      return std::nullopt;
    // A constant selects one physical half; a proved bounded dynamic slot
    // selects a UNION of halves, preserving disjoint A/B residues for any
    // iteration. We never infer that two dynamic same-role slots differ.
    uint32_t slots = 0;
    if (auto c = constant(index.getIndex())) {
      if (*c >= uint64_t(count))
        return std::nullopt;
      slots = 1u << *c;
    } else if (auto mask = index.getIndex().getDefiningOp<arith::AndIOp>()) {
      auto bound = constant(mask.getRhs());
      if (!bound)
        bound = constant(mask.getLhs());
      if (!bound || *bound >= uint64_t(count))
        return std::nullopt;
      // All submasks are in [0,bound]; the full interval is conservative.
      slots = (1u << (*bound + 1)) - 1;
    } else {
      return std::nullopt;
    }
    uint32_t halves = 0;
    for (unsigned i = 0; i < unsigned(count); ++i)
      if (slots & (1u << i))
        halves |= 1u << (2 * i + (enc.getOpIdx() == 0));
    return Region{root, halves, compact};
  }

  static std::optional<Region> resolve(Value source,
                                       ModuleAllocation &allocation) {
    llvm::DenseSet<Value> seen, anchored;
    llvm::DenseMap<Value, SmallVector<Value>> dependents;
    SmallVector<Value> pending{source}, seeds;
    Region result;
    while (!pending.empty()) {
      Value v = pending.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      if (seen.size() > 4096)
        return std::nullopt;
      auto follow = [&](Value input) {
        dependents[input].push_back(v);
        pending.push_back(input);
      };
      if (v.getDefiningOp<Reinterpret>()) {
        auto r = anchor(v, allocation);
        if (!r || (result.owner && (result.owner != r->owner || result.compact != r->compact)))
          return std::nullopt;
        result.owner = r->owner;
        result.compact = r->compact;
        result.halves |= r->halves;
        seeds.push_back(v);
      } else if (auto select = v.getDefiningOp<arith::SelectOp>()) {
        follow(select.getTrueValue());
        follow(select.getFalseValue());
      } else if (auto arg = dyn_cast<BlockArgument>(v)) {
        Block *block = arg.getOwner();
        if (block->hasNoPredecessors())
          return std::nullopt;
        bool incoming = false;
        for (Block *pred : block->getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch || !isa<cf::BranchOp, cf::CondBranchOp>(
                             pred->getTerminator()))
            return std::nullopt;
          for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
            if (branch->getSuccessor(i) != block)
              continue;
            auto ops = branch.getSuccessorOperands(i);
            unsigned produced = ops.getProducedOperandCount();
            unsigned n = arg.getArgNumber();
            if (n < produced || n - produced >= ops.getForwardedOperands().size())
              return std::nullopt;
            follow(ops.getForwardedOperands()[n - produced]);
            incoming = true;
          }
        }
        if (!incoming)
          return std::nullopt;
      } else {
        return std::nullopt;
      }
    }
    while (!seeds.empty()) {
      Value v = seeds.pop_back_val();
      if (!anchored.insert(v).second)
        continue;
      for (Value dependent : dependents[v])
        seeds.push_back(dependent);
    }
    if (!result.owner || !result.halves || anchored.size() != seen.size())
      return std::nullopt;
    return result;
  }

public:
  WholePacketRegions(ModuleOp mod, ModuleAllocation &allocation, StringRef arch) {
    namespace wp = triton::hcugpu::wholepacket;
    // This witness belongs only to this lowering invocation. Imported IR is
    // never authoritative, including on unsupported targets or encodings.
    mod.walk([](Operation *op) {
      op->removeAttr("hcu.packet.internal_physical_halves");
      op->removeAttr("hcu.packet.internal_physical_quarters");
    });
    auto enabled = mod->getAttrOfType<IntegerAttr>("ttg.hcu.whole_packet_v1");
    if (arch != "gfx936" || !enabled || enabled.getInt() != 1 ||
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod) != 64 ||
        triton::gpu::TritonGPUDialect::getNumCTAs(mod) != 1)
      return;
    auto warps = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
    if (!warps)
      return;
    mod.walk([&](Operation *op) {
      Value source;
      if (auto load = dyn_cast<triton::gpu::LocalLoadOp>(op))
        source = load.getSrc();
      else if (auto copy = dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(op))
        source = copy.getDest();
      else
        return;
      auto md = dyn_cast<MD>(source.getType());
      auto enc = md ? dyn_cast<Packet>(md.getEncoding()) : Packet{};
      auto basis = op->getAttrOfType<DenseI32ArrayAttr>("hcu.packet.source_basis");
      if (!enc || !triton::hcugpu::wholepacket::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind()) || md.getRank() != 2 ||
          !basis || basis.size() != 12)
        return;
      std::array<unsigned, 12> b{};
      for (unsigned i = 0; i < b.size(); ++i)
        b[i] = basis[i];
      // This complete bijection proves every logical element maps into the
      // 8192-byte physical packet, independently of its source chart.
      auto semantics=*wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind());
      if (semantics.prefix) {
        if (md.getShape()[0] < 1 || md.getShape()[0] > wp::Elements ||
            md.getShape()[1] < 1 || md.getShape()[1] > wp::Elements ||
            !wp::SourceLayout::getActivePrefix(semantics.storageRows(md.getShape()[0],md.getShape()[1]), semantics.storageColumns(md.getShape()[0],md.getShape()[1]),
                                                warps.getInt(), b))
          return;
      } else if (!wp::SourceLayout::get(md.getShape()[0], md.getShape()[1],
                                        warps.getInt(), b))
        return;
      if (auto r = resolve(source, allocation)) {
        accesses[op] = *r;
        auto offset = r->owner->getAttrOfType<IntegerAttr>("allocation.offset");
        if(r->compact) {
          uint64_t mask=uint64_t(r->halves) << (offset.getInt()/4096);
          assert(mask && mask<=65535 && "checked compact owner contained in 64KiB LDS");
          op->setAttr("hcu.packet.internal_physical_quarters",IntegerAttr::get(IntegerType::get(mod.getContext(),64),mask));
        } else {
          uint64_t mask = uint64_t(r->halves) << (offset.getInt() / wp::Bytes);
          assert(mask && mask <= 255 && "checked owner contained in 64KiB LDS");
          op->setAttr("hcu.packet.internal_physical_halves",
                      IntegerAttr::get(IntegerType::get(mod.getContext(), 64), mask));
        }
      }
    });
  }

  bool disjoint(Operation *before, Operation *after) const {
    auto a = accesses.find(before), b = accesses.find(after);
    return a != accesses.end() && b != accesses.end() &&
           a->second.owner == b->second.owner &&
           !(a->second.halves & b->second.halves);
  }
};

} // namespace mlir::triton::HCU
#endif
