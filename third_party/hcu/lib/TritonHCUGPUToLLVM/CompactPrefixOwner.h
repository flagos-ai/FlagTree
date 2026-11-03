#ifndef TRITON_HCU_COMPACT_PREFIX_OWNER_H
#define TRITON_HCU_COMPACT_PREFIX_OWNER_H
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "Dialect/TritonHCUGPU/IR/WholePacketLayout.h"
#include "WholePacketConsumerPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::HCU {
namespace compact_prefix_detail {
using MD = triton::gpu::MemDescType;
using Alloc = triton::gpu::LocalAllocOp;
using Index = triton::gpu::MemDescIndexOp;
using Slice = triton::gpu::MemDescSubsliceOp;
using View = triton::gpu::MemDescReinterpretOp;
using Packet = triton::gpu::HCUMlsSharedEncodingAttr;
namespace wp = triton::hcugpu::wholepacket;
inline std::optional<int64_t> constant(Value v) {
  auto c=v.getDefiningOp<arith::ConstantOp>();
  auto a=c?dyn_cast<IntegerAttr>(c.getValue()):IntegerAttr{};
  if(!a || a.getValue().getBitWidth()>64 || a.getValue().isNegative())return std::nullopt;
  return a.getInt();
}
inline bool flat(MD t) {
  auto e=dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(t.getEncoding());
  return e && e.getVec()==1 && e.getPerPhase()==1 && e.getMaxPhase()==1 &&
         e.getOrder()==ArrayRef<unsigned>({0});
}
inline bool slot(Value v,int64_t n) {
  if(auto c=constant(v))return *c<n;
  if(auto a=v.getDefiningOp<arith::AndIOp>()) {
    auto c=constant(a.getRhs());if(!c)c=constant(a.getLhs());
    return c && *c<n;
  }
  return false;
}
struct Proof {
  Alloc root;
  SmallVector<Index> indices;
  SmallVector<Slice> slices;
};
inline std::optional<Proof> prove(Alloc root,int64_t warps) {
  auto t=root.getType();
  if(root.getSrc() || !root.isSharedMemoryAlloc() || t.getRank()!=2 ||
     t.getShape()!=t.getAllocShape() || t.getShape()[0]<1 || t.getShape()[0]>4 ||
     t.getShape()[1]!=8192 || !flat(t) ||
     !isa<Float16Type,BFloat16Type>(t.getElementType()) ||
     root.getAlignmentOrDefault()<16384 || root.getAlignmentOrDefault()%16384)
    return std::nullopt;
  Proof p{root,{},{}};SmallVector<Value> pending;llvm::DenseSet<Value> seeds,seen;
  // Only complete flat owner -> index -> slice -> opaque reinterpret chains.
  for(Operation *user:root.getResult().getUsers()) {
    if(auto free=dyn_cast<triton::gpu::LocalDeallocOp>(user)) {
      if(free.getSrc()!=root.getResult())return std::nullopt;
      continue;
    }
    auto index=dyn_cast<Index>(user);
    if(!index || index.getSrc()!=root.getResult() ||
       index.getType().getShape()!=ArrayRef<int64_t>({8192}) ||
       index.getType().getAllocShape()!=index.getType().getShape() ||
       index.getType().getElementType()!=t.getElementType() ||
       index.getType().getMemorySpace()!=t.getMemorySpace() ||
       index.getType().getEncoding()!=t.getEncoding() || !slot(index.getIndex(),t.getShape()[0]))
      return std::nullopt;
    p.indices.push_back(index);
    if(index.getResult().use_empty())return std::nullopt;
    for(Operation *su:index.getResult().getUsers()) {
      auto slice=dyn_cast<Slice>(su);if(!slice || slice.getSrc()!=index.getResult())return std::nullopt;
      auto st=slice.getType();
      if(st.getRank()!=1 || st.getShape()[0]<1 || st.getShape()[0]>2048 ||
         st.getAllocShape()!=index.getType().getShape() || st.getEncoding()!=t.getEncoding() ||
         st.getElementType()!=t.getElementType() || st.getMemorySpace()!=t.getMemorySpace() ||
         slice.getOffsets().size()!=1 || (slice.getOffsets()[0]!=0 && slice.getOffsets()[0]!=4096) ||
         slice.getResult().use_empty())return std::nullopt;
      p.slices.push_back(slice);
      for(Operation *vu:slice.getResult().getUsers()) {
        auto view=dyn_cast<View>(vu);auto vt=view?view.getType():MD{};
        auto enc=vt?dyn_cast<Packet>(vt.getEncoding()):Packet{};
        if(!view || view.getSrc()!=slice.getResult() || !enc ||
           (enc.getVersion()!=257 && enc.getVersion()!=258) || enc.getOpIdx()>1 ||
           enc.getElemBitWidth()!=16 || vt.getRank()!=2 || vt.getAllocShape()!=vt.getShape() ||
           vt.getElementType()!=t.getElementType() || vt.getMemorySpace()!=t.getMemorySpace() ||
           !llvm::equal(enc.getMlsTile(),vt.getShape()) ||
           !llvm::equal(enc.getCTALayout().getCTAsPerCGA(),ArrayRef<unsigned>({1,1})) ||
           !llvm::equal(enc.getCTALayout().getCTASplitNum(),ArrayRef<unsigned>({1,1})) ||
           vt.getShape()[0]<1 || vt.getShape()[1]<1 || vt.getShape()[0]>2048 || vt.getShape()[1]>2048 ||
           vt.getShape()[0]*vt.getShape()[1]!=st.getShape()[0] ||
           slice.getOffsets()[0]!=(enc.getOpIdx()==0?4096:0) ||
           !wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind()))return std::nullopt;
        seeds.insert(view.getResult());pending.push_back(view.getResult());
      }
    }
  }
  if(p.indices.empty() || seeds.empty())return std::nullopt;
  bool producer[2]={false,false},consumer[2]={false,false};
  std::optional<std::array<unsigned,12>> charts[2];std::optional<unsigned> modes[2];
  while(!pending.empty()) {
    Value v=pending.pop_back_val();if(!seen.insert(v).second)continue;
    if(seen.size()>4096)return std::nullopt;
    auto vt=dyn_cast<MD>(v.getType());auto enc=vt?dyn_cast<Packet>(vt.getEncoding()):Packet{};
    if(!enc || (enc.getVersion()!=257 && enc.getVersion()!=258) || enc.getOpIdx()>1)return std::nullopt;
    unsigned role=enc.getOpIdx();
    for(Operation *user:v.getUsers()) {
      if(auto read=dyn_cast<triton::gpu::LocalLoadOp>(user)) {
        if(read.getSrc()!=v || read.getType().getShape()!=vt.getShape())return std::nullopt;
        auto semantics=wp::PacketSemantics::get(enc.getVersion(),role,enc.getAlt2Kind());
        auto basis=read->getAttrOfType<DenseI32ArrayAttr>("hcu.packet.source_basis");
        if(!semantics || !basis || basis.size()!=12)return std::nullopt;
        std::array<unsigned,12> b{};for(unsigned i=0;i<12;++i)b[i]=basis[i];
        auto chart=wp::SourceLayout::getActivePrefix(
            semantics->storageRows(vt.getShape()[0],vt.getShape()[1]),
            semantics->storageColumns(vt.getShape()[0],vt.getShape()[1]),warps,b);
        auto request=root->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
            "ttg.hcu.packet_matrix_read_partition");
        if(!chart || chart->activeElements()>2048)return std::nullopt;
        auto plan=planWholePacketConsumer(read.getType(),*semantics,*chart,role,
            enc.getVersion(),request && request.getInt()==1);
        if(!plan || plan->optimizationBudgetExceeded)return std::nullopt;
        consumer[role]=true;
      } else if(auto copy=dyn_cast<triton::hcugpu::MatrixLoadToLocalOp>(user)) {
        if(copy.getDest()!=v)return std::nullopt;
        producer[role]=true;
      } else if(isa<cf::BranchOp,cf::CondBranchOp>(user)) {
        auto branch=cast<BranchOpInterface>(user);bool forwarded=false;
        for(unsigned s=0;s<branch->getNumSuccessors();++s) {
          auto ops=branch.getSuccessorOperands(s);if(ops.getProducedOperandCount())return std::nullopt;
          auto values=ops.getForwardedOperands();Block *dest=branch->getSuccessor(s);
          for(unsigned i=0;i<values.size();++i)if(values[i]==v) {
            if(i>=dest->getNumArguments() || dest->getArgument(i).getType()!=vt)return std::nullopt;
            pending.push_back(dest->getArgument(i));forwarded=true;
          }
        }
        if(!forwarded)return std::nullopt;
        continue;
      } else return std::nullopt;
      auto basis=user->getAttrOfType<DenseI32ArrayAttr>("hcu.packet.source_basis");
      if(!basis || basis.size()!=12)return std::nullopt;
      std::array<unsigned,12>b{};for(unsigned i=0;i<12;++i)b[i]=basis[i];
      auto sem=wp::PacketSemantics::get(enc.getVersion(),role,enc.getAlt2Kind());if(!sem)return std::nullopt;
      auto chart=wp::SourceLayout::getActivePrefix(sem->storageRows(vt.getShape()[0],vt.getShape()[1]),sem->storageColumns(vt.getShape()[0],vt.getShape()[1]),warps,b);
      if(!chart || chart->activeElements()>2048)return std::nullopt;
      unsigned mode=(sem->transfer==wp::Role::StructuredRows?1:0)|(sem->transpose?2:0);
      if((charts[role] && *charts[role]!=b)||(modes[role] && *modes[role]!=mode))return std::nullopt;
      charts[role]=b;modes[role]=mode;
      for(unsigned u=0;u<chart->activeElements();++u)
        if(wp::physicalHalf(sem->transfer,u)>=2048)return std::nullopt;
    }
  }
  // Forward reachability alone cannot prove mixed-owner CF joins: verify EVERY
  // incoming value is one of this owner's anchored seeds/forwarded arguments.
  for(Value v:seen)if(auto arg=dyn_cast<BlockArgument>(v)) {
    Block *dest=arg.getOwner();if(dest->hasNoPredecessors())return std::nullopt;
    for(Block *pred:dest->getPredecessors()) {
      auto branch=dyn_cast<BranchOpInterface>(pred->getTerminator());
      if(!branch || !isa<cf::BranchOp,cf::CondBranchOp>(pred->getTerminator()))return std::nullopt;
      bool incoming=false;
      for(unsigned s=0;s<branch->getNumSuccessors();++s)if(branch->getSuccessor(s)==dest) {
        auto ops=branch.getSuccessorOperands(s);auto values=ops.getForwardedOperands();
        if(ops.getProducedOperandCount() || arg.getArgNumber()>=values.size() ||
           !seen.contains(values[arg.getArgNumber()]))return std::nullopt;
        incoming=true;
      }
      if(!incoming)return std::nullopt;
    }
  }
  if(!producer[0] || !producer[1] || !consumer[0] || !consumer[1])return std::nullopt;
  return p;
}
inline MD resized(MD t,ArrayRef<int64_t> shape,ArrayRef<int64_t> allocShape) {
  return MD::get(shape,t.getElementType(),t.getEncoding(),t.getMemorySpace(),t.getMutableMemory(),allocShape);
}
} // namespace compact_prefix_detail
// Invoke before ModuleAllocation. Unsupported owners remain byte-for-byte IR
// unchanged. No imported compact witness is trusted, and no CF view type changes.
inline unsigned compactPrefixOwners(ModuleOp mod,StringRef arch) {
  auto enabled=mod->getAttrOfType<BoolAttr>("ttg.hcu.compact_packet_owner");
  if(!enabled || !enabled.getValue())return 0;
  // An imported count is neither proof nor an input. Recompute this invocation.
  mod->setAttr("ttg.hcu.compact_packet_owner_count",IntegerAttr::get(IntegerType::get(mod.getContext(),32),0));
  if(arch!="gfx936" || triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod)!=64 ||
     triton::gpu::TritonGPUDialect::getNumCTAs(mod)!=1)return 0;
  auto warps=mod->getAttrOfType<IntegerAttr>("ttg.num-warps");if(!warps)return 0;
  using namespace compact_prefix_detail;
  SmallVector<Proof> proofs;
  mod.walk([&](Alloc root){if(auto p=prove(root,warps.getInt()))proofs.push_back(*p);});
  // All evidence has been collected before ANY root/type/offset mutation.
  for(auto &p:proofs) {
    SmallVector<int64_t> shape(p.root.getType().getShape());shape[1]=4096;
    p.root.getResult().setType(resized(p.root.getType(),shape,shape));
    for(auto index:p.indices)index.getResult().setType(resized(index.getType(),{4096},{4096}));
    for(auto slice:p.slices) {
      slice.getResult().setType(resized(slice.getType(),slice.getType().getShape(),{4096}));
      if(slice.getOffsets()[0]==4096)slice->setAttr("offsets",DenseI32ArrayAttr::get(mod.getContext(),{2048}));
    }
  }
  mod->setAttr("ttg.hcu.compact_packet_owner_count",IntegerAttr::get(IntegerType::get(mod.getContext(),32),proofs.size()));
  return proofs.size();
}
} // namespace mlir::triton::HCU
#endif
