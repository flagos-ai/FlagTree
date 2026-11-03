#include "WholePacketConsumerPlan.h"
// Opt-in experimental encoding version256. This source chart payload remains
// explicit on both typed producer/consumer operations until it becomes a TD
// parameter. No physical LinearLayout is ever synthesized.
namespace {
namespace wp = mlir::triton::hcugpu::wholepacket;
constexpr unsigned WholePacketVersion = 256;
constexpr unsigned PacketPrefixVersion = 257;
static bool isPacketVersion(unsigned version) {
  return version == WholePacketVersion || version == PacketPrefixVersion || version == 258;
}
static bool isWholePacket(Type t) {
  auto m = dyn_cast<MemDescType>(t);
  auto e = m ? dyn_cast<HCUMlsSharedEncodingAttr>(m.getEncoding()) : HCUMlsSharedEncodingAttr{};
  return e && isPacketVersion(e.getVersion());
}
static std::optional<wp::SourceLayout> packetLayout(Operation *op, MemDescType t) {
  auto enc = dyn_cast<HCUMlsSharedEncodingAttr>(t.getEncoding());
  auto attr = op->getAttrOfType<DenseI32ArrayAttr>("hcu.packet.source_basis");
  if (!enc || !isPacketVersion(enc.getVersion()) || !attr || attr.size()!=12 ||
      t.getRank()!=2 || t.getElementType().getIntOrFloatBitWidth()!=16 ||
      enc.getElemBitWidth()!=16 || !wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind()) ||
      !llvm::equal(enc.getMlsTile(),t.getShape()) ||
      !llvm::equal(enc.getCTALayout().getCTAsPerCGA(),ArrayRef<unsigned>({1,1})) ||
      !llvm::equal(enc.getCTALayout().getCTASplitNum(),ArrayRef<unsigned>({1,1})))
    return std::nullopt;
  std::array<unsigned,12> bases{};
  for(unsigned i=0;i<12;++i) bases[i]=attr[i];
  auto warps=op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>("ttg.num-warps");
  if(!warps)return std::nullopt;
  auto semantics=*wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind());
  if(semantics.prefix) {
    if(t.getShape()[0]<1 || t.getShape()[0]>wp::Elements ||
       t.getShape()[1]<1 || t.getShape()[1]>wp::Elements)return std::nullopt;
    return wp::SourceLayout::getActivePrefix(semantics.storageRows(t.getShape()[0],t.getShape()[1]),semantics.storageColumns(t.getShape()[0],t.getShape()[1]),warps.getInt(),bases);
  }
  return wp::SourceLayout::get(t.getShape()[0],t.getShape()[1],warps.getInt(),bases);
}
struct PacketView { LocalAllocOp owner; Value slot; unsigned halfOffset; };
static std::optional<PacketView> packetView(Value v) {
  auto view=v.getDefiningOp<MemDescReinterpretOp>();
  auto enc=view ? dyn_cast<HCUMlsSharedEncodingAttr>(view.getType().getEncoding()) : HCUMlsSharedEncodingAttr{};
  if(!enc || !isPacketVersion(enc.getVersion()))return std::nullopt;
  int64_t active=4096;
  if(enc.getVersion()!=WholePacketVersion) {
    auto shape=view.getType().getShape();
    if(shape.size()!=2 || shape[0]<1 || shape[0]>4096 || shape[1]<1 || shape[1]>4096 ||
       shape[0]*shape[1]>4096)return std::nullopt;
    active=shape[0]*shape[1];
  }
  auto slice=view ? view.getSrc().getDefiningOp<MemDescSubsliceOp>() : MemDescSubsliceOp{};
  auto index=slice ? slice.getSrc().getDefiningOp<MemDescIndexOp>() : MemDescIndexOp{};
  auto root=index ? index.getSrc().getDefiningOp<LocalAllocOp>() : LocalAllocOp{};
  bool compact=root && root.getType().getRank()==2 && root.getType().getShape()[1]==4096;
  unsigned halfElements=compact?2048:4096, frameElements=2*halfElements;
  auto compactEnabled=view->getParentOfType<ModuleOp>()->getAttrOfType<BoolAttr>("ttg.hcu.compact_packet_owner");
  if(compact && (!compactEnabled || !compactEnabled.getValue() || enc.getVersion()==WholePacketVersion || active>2048 ||
      root.getType().getAllocShape()!=root.getType().getShape() || index.getType().getAllocShape()!=index.getType().getShape() ||
      slice.getType().getAllocShape()!=index.getType().getShape() || view.getType().getAllocShape()!=view.getType().getShape()))return std::nullopt;
  if(!root || root.getSrc() || root.getType().getRank()!=2 ||
      root.getType().getShape()[1]!=frameElements || root.getAlignmentOrDefault()<16384 ||
      !isa<SwizzledSharedEncodingAttr>(root.getType().getEncoding()) ||
      slice.getOffsets().size()!=1 || slice.getType().getShape()!=ArrayRef<int64_t>({active}) ||
      index.getType().getShape()!=ArrayRef<int64_t>({int64_t(frameElements)}) ||
      root.getType().getElementType()!=view.getType().getElementType())return std::nullopt;
  auto shared=cast<SwizzledSharedEncodingAttr>(root.getType().getEncoding());
  if(shared.getVec()!=1 || shared.getPerPhase()!=1 || shared.getMaxPhase()!=1 ||
      shared.getOrder()!=ArrayRef<unsigned>({0}))return std::nullopt;
  unsigned slots=root.getType().getShape()[0];
  if(compact ? (!slots || uint64_t(slots)*8192>65536) : !wp::WholeFrame::get(slots,65536))return std::nullopt;
  auto constant=getSignedI64Constant(index.getIndex());
  bool slotProven=constant && *constant>=0 && *constant<slots;
  if(auto mask=index.getIndex().getDefiningOp<arith::AndIOp>()) {
    auto bound=getSignedI64Constant(mask.getRhs());
    slotProven|=bound && *bound>=0 && uint64_t(*bound)<slots;
  }
  if(!slotProven)return std::nullopt;
  unsigned off=slice.getOffsets()[0];
  if(off!=0 && off!=halfElements)return std::nullopt;
  if(off!=(enc.getOpIdx()==0?halfElements:0))return std::nullopt;
  return PacketView{root,index.getIndex(),off};
}
// CF queue values are accepted only when every incoming edge resolves to a
// complete reinterpret of the same physical owner/role. Cycles contribute no
// authority: at least one concrete seed must be reached and all edge kinds are
// checked. No entry argument, subview, select or unknown producer is admitted.
static bool collectPacketViews(Value v, llvm::SmallPtrSetImpl<void *> &seen,
                               SmallVectorImpl<Value> &seeds) {
  if(!seen.insert(v.getAsOpaquePointer()).second)return true;
  if(v.getDefiningOp<MemDescReinterpretOp>()) {
    if(!packetView(v))return false;
    seeds.push_back(v);return true;
  }
  auto arg=dyn_cast<BlockArgument>(v);
  if(!arg || arg.getOwner()->hasNoPredecessors())return false;
  Block *dest=arg.getOwner(); unsigned n=arg.getArgNumber();
  for(Block *pred:dest->getPredecessors()) {
    Operation *term=pred->getTerminator(); bool edge=false;
    if(auto br=dyn_cast<cf::BranchOp>(term)) {
      if(br.getDest()!=dest || n>=br.getDestOperands().size() ||
         !collectPacketViews(br.getDestOperands()[n],seen,seeds))return false;
      edge=true;
    } else if(auto br=dyn_cast<cf::CondBranchOp>(term)) {
      if(br.getTrueDest()==dest) {
        if(n>=br.getTrueDestOperands().size() || !collectPacketViews(br.getTrueDestOperands()[n],seen,seeds))return false;
        edge=true;
      }
      if(br.getFalseDest()==dest) {
        if(n>=br.getFalseDestOperands().size() || !collectPacketViews(br.getFalseDestOperands()[n],seen,seeds))return false;
        edge=true;
      }
    }
    if(!edge)return false;
  }
  return true;
}
static std::optional<PacketView> packetQueueView(Value v) {
  llvm::SmallPtrSet<void *,32> seen;SmallVector<Value> seeds;
  if(!collectPacketViews(v,seen,seeds) || seeds.empty())return std::nullopt;
  auto first=packetView(seeds.front());
  for(Value seed:seeds) {
    auto next=packetView(seed);
    if(!next || next->owner!=first->owner || next->halfOffset!=first->halfOffset)return std::nullopt;
  }
  return first;
}
// Scalar SSA is not automatically CTA-uniform: CF PHIs also require uniform
// control. Reject every unsupported leaf, including thread/lane IDs and loads.
static bool packetUniformValue(Value v,llvm::SmallPtrSetImpl<void *> &seen) {
  if(!v.getType().isIntOrIndex())return false;
  if(!seen.insert(v.getAsOpaquePointer()).second)return true;
  if(auto arg=dyn_cast<BlockArgument>(v)) {
    Block *dest=arg.getOwner();unsigned n=arg.getArgNumber();
    if(auto f=dyn_cast<triton::FuncOp>(dest->getParentOp()))
      if(dest==&f.getBody().front())return true; // scalar kernel ABI
    if(dest->hasNoPredecessors())return false;
    for(Block *pred:dest->getPredecessors()) {
      Operation *t=pred->getTerminator();bool edge=false;
      if(auto br=dyn_cast<cf::BranchOp>(t)) {
        if(br.getDest()!=dest || !packetUniformValue(br.getDestOperands()[n],seen))return false;edge=true;
      } else if(auto br=dyn_cast<cf::CondBranchOp>(t)) {
        if(!packetUniformValue(br.getCondition(),seen))return false;
        if(br.getTrueDest()==dest) {if(!packetUniformValue(br.getTrueDestOperands()[n],seen))return false;edge=true;}
        if(br.getFalseDest()==dest) {if(!packetUniformValue(br.getFalseDestOperands()[n],seen))return false;edge=true;}
      }
      if(!edge)return false;
    }
    return true;
  }
  Operation *op=v.getDefiningOp();if(!op)return false;
  if(isa<triton::GetProgramIdOp>(op))return true;
  if(!op->getName().getStringRef().starts_with("arith.") || op->getNumRegions())return false;
  for(Value operand:op->getOperands())if(!packetUniformValue(operand,seen))return false;
  return true;
}
static bool packetUniform(Value v) {
  llvm::SmallPtrSet<void *,32> seen;
  if(!packetUniformValue(v,seen))return false;
  auto f=v.getParentBlock()->getParentOp()->getParentOfType<triton::FuncOp>();
  if(!f)f=dyn_cast<triton::FuncOp>(v.getParentBlock()->getParentOp());
  if(!f)return false;
  bool result=true;f.walk([&](cf::CondBranchOp br) {
    llvm::SmallPtrSet<void *,32> control;
    result&=packetUniformValue(br.getCondition(),control);
  });
  return result;
}
// Power-of-two divisibility is invariant under integer wrap, so it does
// not require nowrap assumptions. CF cycles must retain a divisible entry seed.
static bool packetMultiple(Value v,unsigned n,llvm::SmallPtrSetImpl<void *> &active) {
  if(!v.getType().isInteger() || !n || (n&(n-1)))return false;
  if(auto c=getSignedI64Constant(v))return uint64_t(*c)%n==0;
  if(!active.insert(v.getAsOpaquePointer()).second)return true;
  bool answer=[&]() {
    if(auto arg=dyn_cast<BlockArgument>(v)) {
      Block *dest=arg.getOwner();unsigned k=arg.getArgNumber();if(dest->hasNoPredecessors())return false;
      for(Block *pred:dest->getPredecessors()) {
        Operation *t=pred->getTerminator();bool edge=false;
        if(auto br=dyn_cast<cf::BranchOp>(t)) {
          if(br.getDest()!=dest || !packetMultiple(br.getDestOperands()[k],n,active))return false;edge=true;
        } else if(auto br=dyn_cast<cf::CondBranchOp>(t)) {
          if(br.getTrueDest()==dest) {if(!packetMultiple(br.getTrueDestOperands()[k],n,active))return false;edge=true;}
          if(br.getFalseDest()==dest) {if(!packetMultiple(br.getFalseDestOperands()[k],n,active))return false;edge=true;}
        }
        if(!edge)return false;
      }
      return true;
    }
    Operation *op=v.getDefiningOp();if(!op)return false;
    auto multiple=[&](Value x){return packetMultiple(x,n,active);};
    if(isa<arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp>(op)) {
      auto src=op->getOperand(0);auto srcTy=dyn_cast<IntegerType>(src.getType());auto dstTy=dyn_cast<IntegerType>(v.getType());
      return srcTy && dstTy && uint64_t(n)<=(uint64_t(1)<<std::min<unsigned>(31,std::min(srcTy.getWidth(),dstTy.getWidth()))) && multiple(src);
    }
    if(isa<arith::AddIOp,arith::SubIOp,arith::OrIOp,arith::XOrIOp,arith::MaxSIOp,arith::MinSIOp,arith::MaxUIOp,arith::MinUIOp>(op))
      return multiple(op->getOperand(0)) && multiple(op->getOperand(1));
    if(auto mul=dyn_cast<arith::MulIOp>(op))return multiple(mul.getLhs()) || multiple(mul.getRhs());
    if(isa<arith::RemSIOp,arith::RemUIOp>(op)) {
      auto c=getSignedI64Constant(op->getOperand(1));return c && *c>0 && uint64_t(*c)%n==0 && multiple(op->getOperand(0));
    }
    if(auto select=dyn_cast<arith::SelectOp>(op))return multiple(select.getTrueValue()) && multiple(select.getFalseValue());
    if(auto bit=dyn_cast<arith::AndIOp>(op)) {
      auto c=getSignedI64Constant(bit.getRhs());return (c && (uint64_t(*c)&(n-1))==0) || multiple(bit.getLhs()) || multiple(bit.getRhs());
    }
    return false;
  }();
  active.erase(v.getAsOpaquePointer());return answer;
}
static bool packetMultiple(Value v,unsigned n) {llvm::SmallPtrSet<void *,32> active;return packetMultiple(v,n,active);}
static Value packetUniformSaturatingSub(Location loc,ConversionPatternRewriter &rw,Value lhs,Value rhs) {
  return LLVM::InlineAsmOp::create(rw,loc,lhs.getType(),ValueRange{lhs,rhs},
      "s_sub_u32 $0, $1, $2\n\ts_cmp_ge_u32 $1, $2\n\ts_cselect_b32 $0, $0, 0",
      "=&s,s,s,~{scc}",false,false,LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(rw.getContext(),LLVM::AsmDialect::AD_ATT),ArrayAttr::get(rw.getContext(),{})).getRes();
}
#include "PacketIndexBounds.inc"

static LogicalResult verifyWholePackets(ModuleOp mod,StringRef arch) {
  // Imported attributes are never proof. Clear every internal witness before
  // validating this exact module; only the successful preflight below writes
  // them, and producer lowering consumes them in this same conversion pass.
  mod.walk([&](triton::hcugpu::MatrixLoadToLocalOp op) {
    op->removeAttr("hcu.packet.internal_verified_shape");
    op->removeAttr("hcu.packet.internal_verified_column");
    op->removeAttr("hcu.packet.internal_verified_column_bounds");
    op->removeAttr("hcu.packet.internal_verified_full_rows");
  });
  bool found=false;mod.walk([&](Operation *op){for(Type t:op->getOperandTypes())found|=isWholePacket(t);for(Type t:op->getResultTypes())found|=isWholePacket(t);});
  if(!found)return success();
  auto fail=[&](StringRef s){return mod.emitError("wholepacket-v1: ")<<s;};
  auto enabled=mod->getAttrOfType<IntegerAttr>("ttg.hcu.whole_packet_v1");
  if(!enabled || enabled.getInt()!=1 || arch!="gfx936" ||
      TritonGPUDialect::getThreadsPerWarp(mod)!=64 || TritonGPUDialect::getNumCTAs(mod)!=1)
    return fail("requires explicit opt-in, gfx936 wave64 single CTA");
  LogicalResult result=success();
  // Both producer and consumer metadata must agree for a complete owner/role,
  // including views that travel through rotating CF SSA queues.
  SmallVector<std::tuple<Operation *,unsigned,unsigned,std::array<unsigned,12>>> charts;
  mod.walk([&](Operation *op) {
    if(failed(result))return;
    for(Value v:op->getResults())if(isWholePacket(v.getType()) && !v.getDefiningOp<MemDescReinterpretOp>()) {
      result=fail("opaque values must originate in a complete whole-frame reinterpret");return;
    }
    for(Value v:op->getOperands())if(isWholePacket(v.getType())) {
      if(!packetQueueView(v)) {result=fail("unsupported opaque block argument or unproved view origin");return;}
      if(!isa<triton::hcugpu::MatrixLoadToLocalOp,LocalLoadOp,cf::BranchOp,cf::CondBranchOp>(op)) {
        result=fail("unsupported opaque subview/use");return;
      }
    }
  });
  auto checkChart=[&](Operation *op,Value value)->std::optional<wp::SourceLayout> {
    auto owner=packetQueueView(value);auto t=cast<MemDescType>(value.getType());
    auto p=packetLayout(op,t);
    if(!owner || !p)return std::nullopt;
    unsigned role=cast<HCUMlsSharedEncodingAttr>(t.getEncoding()).getOpIdx();
    auto enc=cast<HCUMlsSharedEncodingAttr>(t.getEncoding());
    auto semantics=*wp::PacketSemantics::get(enc.getVersion(),role,enc.getAlt2Kind());
    unsigned mode=(semantics.transfer==wp::Role::StructuredRows?1u:0u) |
                  (semantics.transpose?2u:0u);
    // Preserve compatible v256/v257 full-packet views. Compare the effective
    // transfer/order contract, rather than requiring identical version tags.
    for(auto &[root,r,previousMode,basis]:charts)
      if(root==owner->owner.getOperation() && r==role &&
         (basis!=p->basis || previousMode!=mode))return std::nullopt;
    charts.emplace_back(owner->owner.getOperation(),role,mode,p->basis);return p;
  };
  mod.walk([&](LocalLoadOp read) {
    if(failed(result) || !isWholePacket(read.getSrc().getType()))return;
    if(!checkChart(read,read.getSrc()) || read.getType().getShape()!=read.getSrc().getType().getShape() ||
       !isa<DotOperandEncodingAttr,LinearEncodingAttr>(read.getType().getEncoding()))
      result=fail("producer/consumer source chart or logical shape mismatch");
  });
  mod.walk([&](triton::hcugpu::MatrixLoadToLocalOp producer) {
    if(failed(result) || !isWholePacket(producer.getDest().getType()))return;
    auto p=checkChart(producer,producer.getDest());
    if(!p){result=fail("missing producer/consumer or invalid bijective vector source chart");return;}
    if(producer.getCache()!=triton::CacheModifier::NONE || producer.getEvict()!=triton::EvictionPolicy::NORMAL || producer.getIsVolatile()) {
      result=fail("non-default cache/eviction or volatile packet copy is unsupported");return;
    }
    auto pointer=dyn_cast<triton::PointerType>(producer.getBase().getType());
    if(!pointer || pointer.getAddressSpace()!=1 || pointer.getPointeeType()!=producer.getDest().getType().getElementType()) {
      result=fail("packet source requires matching AS1 element pointer");return;
    }
    auto shape=producer.getShape(),stride=producer.getStrides(),index=producer.getIndices();
    if(shape.size()!=2 || stride.size()!=2 || index.size()!=3 ||
       !llvm::equal(producer.getTensorShape(),producer.getDest().getType().getShape())) {
      result=fail("packet copy requires rank-two shape/strides and row, column, enable indices");return;
    }
    auto r=getSignedI64Constant(shape[0]),c=getSignedI64Constant(shape[1]);
    if(!r || !c || *r<1 || *c<1 || uint64_t(*r)>0x7fffffffull/(2ull*uint64_t(*c)) ||
       getSignedI64Constant(stride[0])!=*c || getSignedI64Constant(stride[1])!=1) {
      result=fail("unknown/overflowing true source extent or non-dense stride");return;
    }
    auto enc=cast<HCUMlsSharedEncodingAttr>(producer.getDest().getType().getEncoding());
    bool structured=wp::PacketSemantics::get(enc.getVersion(),enc.getOpIdx(),enc.getAlt2Kind())->transfer==wp::Role::StructuredRows;
    if(structured ? !wp::StructuredDomain::get(*c,*r) : *c%16) {
      result=fail("source indices/structured mode/vector boundary not proven");return;
    }
    {
      // Explicit bounded producer API: index[2] is an I32 boolean predicate;
      // whole-resource descriptors and signed-I64 checks preserve OOB-zero
      // semantics for arbitrary I32 tile origins, without an extent assumption.
      auto global=mod->getAttrOfType<IntegerAttr>("ttg.hcu.whole_packet_global_bounds");
      if(!global || global.getInt()!=1 || !index[2].getType().isInteger(32)) {
        result=fail("bounded producer requires explicit global-bounds API");return;
      }
    }
    if(!packetUniform(index[0]) || !packetUniform(index[1]) || !packetUniform(index[2])) {
      result=fail("bounded producer indices/predicate require CTA-uniform SSA and control");return;
    }
    auto arg=dyn_cast<BlockArgument>(producer.getBase());auto f=producer->getParentOfType<triton::FuncOp>();
    auto alignment=arg?f.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),"tt.divisibility"):IntegerAttr{};
    if(!arg || arg.getOwner()!=&f.getBody().front() || !alignment || alignment.getInt()<16 || alignment.getInt()%16) {
      result=fail("global source base requires kernel argument alignment16 proof");return;
    }
    {
      producer->setAttr("hcu.packet.internal_verified_shape",DenseI64ArrayAttr::get(mod.getContext(),ArrayRef<int64_t>{*r,*c}));
      bool wholeColumn=*c%p->columns==0 && packetMultiple(index[1],p->columns);
      producer->setAttr("hcu.packet.internal_verified_column",BoolAttr::get(mod.getContext(),wholeColumn));
      auto rowRange=packetIndexBounds(index[0],producer);
      auto columnRange=packetIndexBounds(index[1],producer);
      bool columnInBounds=wholeColumn && columnRange && columnRange->lo>=0 &&
          __int128(columnRange->hi)+p->columns<=*c;
      bool fullRows=rowRange && rowRange->lo>=0 &&
          __int128(rowRange->hi)+p->rows<=*r;
      producer->setAttr("hcu.packet.internal_verified_column_bounds",BoolAttr::get(mod.getContext(),columnInBounds));
      producer->setAttr("hcu.packet.internal_verified_full_rows",BoolAttr::get(mod.getContext(),fullRows));
    }
  });
  return result;
}
static Value packetLinear(TritonLLVMOpBuilder &b,Value x,const std::array<unsigned,12>&basis) {
  Value result=b.i32_val(0);
  for(unsigned bit=0;bit<12;++bit)if(basis[bit])
    result=b.xor_(result,b.mul(b.and_(b.lshr(x,b.i32_val(bit)),b.i32_val(1)),b.i32_val(basis[bit])));
  return result;
}
static Value packetPhysical(TritonLLVMOpBuilder &b,Value logical,const wp::SourceLayout &p,bool structured) {
  Value u=packetLinear(b,logical,p.inverseBasis),pw=b.lshr(u,b.i32_val(9));
  Value rotation=b.mul(b.and_(pw,b.i32_val(structured?3:1)),b.i32_val(structured?8:32));
  return b.or_(b.and_(u,b.i32_val(0xfffffe00u)),b.and_(b.add(b.and_(u,b.i32_val(511)),rotation),b.i32_val(511)));
}
// Physical LDS scopes are facts supplied by the same-pass core allocation
// proof, not by the imported opaque layout. Core clears all imported witnesses
// before recomputing them. A union mask is conservative: consumers exclude only
// halves outside the entire accessed union, never another access to that union.
static void packetPhysicalAliasScopes(Operation *source, Operation *lowered,
                                      bool consumer) {
  // Compact-owner witnesses use independent absolute 4KiB cells. The legacy
  // half-domain path below is intentionally unchanged.
  if(auto q=source->getAttrOfType<IntegerAttr>("hcu.packet.internal_physical_quarters")) {
    if(!q.getType().isInteger(64))return;
    uint64_t mask=q.getValue().getZExtValue();if(!mask || mask>65535)return;
    auto iface=dyn_cast<LLVM::AliasAnalysisOpInterface>(lowered);if(!iface)return;
    Builder b(source->getContext());
    auto domain=b.getAttr<LLVM::AliasScopeDomainAttr>(b.getStringAttr("hcu.packet.physical.lds.quarters"),b.getStringAttr("Proven absolute physical LDS 4KiB quarter regions"));
    SmallVector<Attribute> accessed,excluded;
    for(unsigned cell=0;cell<16;++cell) {
      auto scope=b.getAttr<LLVM::AliasScopeAttr>(b.getStringAttr("hcu.packet.physical.lds.quarter"+std::to_string(cell)),domain,
        b.getStringAttr("Absolute LDS bytes["+std::to_string(cell*4096)+","+std::to_string((cell+1)*4096)+")"));
      if(mask&(uint64_t(1)<<cell))accessed.push_back(scope);else excluded.push_back(scope);
    }
    iface.setAliasScopes(b.getArrayAttr(accessed));
    if(consumer && !excluded.empty())iface.setNoAliasScopes(b.getArrayAttr(excluded));
    return;
  }
  auto attr=source->getAttrOfType<IntegerAttr>("hcu.packet.internal_physical_halves");
  if(!attr || !attr.getType().isInteger(64))return;
  uint64_t mask=attr.getValue().getZExtValue();
  if(!mask || mask>255)return;
  auto iface=dyn_cast<LLVM::AliasAnalysisOpInterface>(lowered);
  if(!iface)return;
  Builder b(source->getContext());
  auto domain=b.getAttr<LLVM::AliasScopeDomainAttr>(
      b.getStringAttr("hcu.packet.physical.lds"),
      b.getStringAttr("Proven absolute physical LDS 8KiB half regions"));
  SmallVector<Attribute> accessed,excluded;
  for(unsigned half=0;half<8;++half) {
    auto scope=b.getAttr<LLVM::AliasScopeAttr>(
        b.getStringAttr("hcu.packet.physical.lds.half"+std::to_string(half)),
        domain,b.getStringAttr("Absolute LDS bytes["+std::to_string(half*8192)+
                              ","+std::to_string((half+1)*8192)+")"));
    if(mask&(uint64_t(1)<<half))accessed.push_back(scope);
    else excluded.push_back(scope);
  }
  iface.setAliasScopes(b.getArrayAttr(accessed));
  if(consumer && !excluded.empty())
    iface.setNoAliasScopes(b.getArrayAttr(excluded));
}
static LogicalResult lowerWholePacketProducer(triton::hcugpu::MatrixLoadToLocalOp op,
    triton::hcugpu::MatrixLoadToLocalOpAdaptor a,const LLVMTypeConverter &converter,
    ConversionPatternRewriter &rw) {
  auto p=packetLayout(op,op.getDest().getType());if(!p)return op.emitError("wholepacket: invalid source chart");
  auto enc=cast<HCUMlsSharedEncodingAttr>(op.getDest().getType().getEncoding());unsigned role=enc.getOpIdx();
  bool structured=wp::PacketSemantics::get(enc.getVersion(),role,enc.getAlt2Kind())->transfer==wp::Role::StructuredRows;
  auto &rewriter=rw;
  auto loc=op.getLoc();auto b=TritonLLVMOpBuilder(loc,rw);auto *ctx=rw.getContext();
  auto elem=converter.convertType(op.getDest().getType().getElementType());
  auto smem=getSharedMemoryObjectFromStruct(loc,a.getDest(),elem,rw);
  // Hardware lane/wave identity is invariant for the whole kernel. Materialize
  // its convergent readfirstlane at function entry, before any divergent CFG or
  // wait/publish boundary, rather than asking LLVM LICM to speculate a
  // convergent call out of the producer loop. Every use is dominated by entry;
  // ordinary CSE merges the identical entry-local computations. No imported
  // attribute or call-site execution-mask assumption is used.
  Value lane,wave;
  {
    OpBuilder::InsertionGuard guard(rw);
    if(auto fn=op->getParentOfType<LLVM::LLVMFuncOp>())
      rw.setInsertionPointToStart(&fn.getBody().front());
    else if(auto fn=op->getParentOfType<triton::FuncOp>())
      rw.setInsertionPointToStart(&fn.getBody().front());
    auto ids=getLaneAndWarpId(rw,loc);lane=ids.first;wave=ids.second;
    wave=rw.create<ROCDL::ReadfirstlaneOp>(loc,i32_ty,wave);
  }
  bool global=a.getIndices().size()==3;
  auto shapeProof=op->getAttrOfType<DenseI64ArrayAttr>("hcu.packet.internal_verified_shape");
  auto columnProof=op->getAttrOfType<BoolAttr>("hcu.packet.internal_verified_column");
  if(global && (!shapeProof || shapeProof.size()!=2 || !columnProof))
    return op.emitError("wholepacket: missing same-pass preflight source witness");
  std::optional<int64_t> sourceCols=global?std::optional<int64_t>(shapeProof[1]):getSignedI64Constant(op.getShape()[1]);
  bool wholeColumn=global && columnProof.getValue();
  auto columnBounds=op->getAttrOfType<BoolAttr>("hcu.packet.internal_verified_column_bounds");
  auto fullRows=op->getAttrOfType<BoolAttr>("hcu.packet.internal_verified_full_rows");
  bool columnInBounds=wholeColumn && columnBounds && columnBounds.getValue();
  bool rowInBounds=global && fullRows && fullRows.getValue();
  op->removeAttr("hcu.packet.internal_verified_shape");
  op->removeAttr("hcu.packet.internal_verified_column");
  op->removeAttr("hcu.packet.internal_verified_column_bounds");
  op->removeAttr("hcu.packet.internal_verified_full_rows");
  uint64_t localBytes=sourceCols?2ull*((p->rows-1)*uint64_t(*sourceCols)+(p->columns-8)):0x80000000ull;
  bool boundedRaw=global && !structured && localBytes<0x80000000ull;
  bool provenRawRows=boundedRaw && wholeColumn && rowInBounds;
  Value uniformColumnValid=b.i1_val(true);
  if(wholeColumn && !columnInBounds)uniformColumnValid=b.and_(b.icmp_sge(a.getIndices()[1],b.i32_val(0)),b.icmp_ult(a.getIndices()[1],a.getShape()[1]));
  Value rowStart=global?Value(b.sext(i64_ty,a.getIndices()[0])):Value(b.zext(i64_ty,a.getIndices()[0]));
  Value colStart=global?Value(b.sext(i64_ty,a.getIndices()[1])):Value(b.zext(i64_ty,a.getIndices()[1]));
  Value start=b.add(b.mul(rowStart,a.getStrides()[0]),colStart);
  Value rawUniform=b.shl(start,b.i64_val(1));
  Value rebase=(global && !provenRawRows)?b.select(b.icmp_sge(rawUniform,b.i64_val(0)),rawUniform,b.i64_val(0)):rawUniform;
  Value byteStart=rebase;if(structured)byteStart=b.sub(byteStart,b.i64_val(16));
  Value base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),byteStart);
  Value mode=structured?b.trunc(i16_ty,b.or_(b.shl(a.getShape()[1],b.i32_val(1)),b.i32_val(0x4000))):b.int_val(16,0);
  Value records;
  if(!global) {
    if(structured)records=b.sub(a.getShape()[0],a.getIndices()[0]);
    else records=b.trunc(i32_ty,b.shl(b.sub(b.mul(b.zext(i64_ty,a.getShape()[0]),b.zext(i64_ty,a.getShape()[1])),start),b.i64_val(1)));
  } else {
    // A resource keeps the exact full row count. Rebasing A in both row and K
    // changes row semantics, so leave its full-resource mode until a separate
    // row-origin proof can shrink it. Raw B uses byte bounds and SALU clamp.
    if(structured) {
      if(wholeColumn) {
        Value rowRebase=b.select(b.icmp_sge(a.getIndices()[0],b.i32_val(0)),a.getIndices()[0],b.i32_val(0));
        Value rowBytes=b.shl(b.mul(b.zext(i64_ty,rowRebase),a.getStrides()[0]),b.i64_val(1));
        Value origin=b.sub(b.add(rowBytes,b.shl(colStart,b.i64_val(1))),b.i64_val(16));
        base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),origin);
        Value remaining=packetUniformSaturatingSub(loc,rw,a.getShape()[0],rowRebase);
        Value valid=b.and_(uniformColumnValid,b.icmp_ne(a.getIndices()[2],b.i32_val(0)));
        records=b.select(valid,remaining,b.i32_val(0));
      } else {
        byteStart=b.i64_val(uint64_t(-16));base=b.gep(ptr_ty(ctx,1),i8_ty,a.getBase(),byteStart);
        records=a.getShape()[0];
      }
    } else if(provenRawRows) {
      // A full row tile plus the existing complete-column predicate proves
      // 0<=origin<trueBytes whenever enabled. Use the tile-local byte span
      // when representable; otherwise retain the full-resource remainder.
      // Invalid columns/predicates still perform an OOB-zero LDS write.
      // No LLVM assume, unchecked source load or skipped write.
      Value enabled=b.and_(uniformColumnValid,b.icmp_ne(a.getIndices()[2],b.i32_val(0)));
      uint64_t packetSpan=localBytes+16;
      if(packetSpan<0x80000000ull) {
        // Complete rows and tile-aligned complete columns prove every emitted
        // 16B chunk lies in both the true resource and this local rectangle.
        // The descriptor cap is independent of the moving tile origin; holes
        // between rows are never requested. Disabled requests still zero-fill.
        records=b.select(enabled,b.i32_val(packetSpan),b.i32_val(0));
      } else {
        Value bytes=b.shl(b.mul(b.zext(i64_ty,a.getShape()[0]),b.zext(i64_ty,a.getShape()[1])),b.i64_val(1));
        Value remaining=b.sub(b.trunc(i32_ty,bytes),b.trunc(i32_ty,rawUniform));
        records=b.select(enabled,remaining,b.i32_val(0));
      }
    } else {
      Value bytes=b.shl(b.mul(b.zext(i64_ty,a.getShape()[0]),b.zext(i64_ty,a.getShape()[1])),b.i64_val(1));
      Value fits=b.icmp_ule(rebase,bytes);
      Value saturated=packetUniformSaturatingSub(loc,rw,b.trunc(i32_ty,bytes),b.trunc(i32_ty,rebase));
      Value canReach=b.icmp_sge(rawUniform,b.i64_val(-int64_t(localBytes)));
      Value enabled=b.and_(fits,b.and_(canReach,b.icmp_ne(a.getIndices()[2],b.i32_val(0))));
      if(wholeColumn)enabled=b.and_(enabled,uniformColumnValid);
      records=b.select(enabled,saturated,b.i32_val(0));
    }
  }
  Value resource=rw.createOrFold<ROCDL::MakeBufferRsrcOp>(loc,ptr_ty(ctx,8),base,mode,records,b.i32_val(0x20000));
  auto fn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rw,
      structured?"llvm.amdgcn.struct.ptr.buffer.load.lds":"llvm.amdgcn.raw.ptr.buffer.load.lds",
      LLVM::LLVMVoidType::get(ctx),structured?ArrayRef<Type>{ptr_ty(ctx,8),ptr_ty(ctx,3),i32_ty,i32_ty,i32_ty,i32_ty,i32_ty,i32_ty}:ArrayRef<Type>{ptr_ty(ctx,8),ptr_ty(ctx,3),i32_ty,i32_ty,i32_ty,i32_ty,i32_ty});
  for(unsigned copy=0;copy<p->copiesPerWave();++copy) {
    Value pw=b.add(wave,b.i32_val(copy*p->warps));
    Value ordinal=b.add(b.shl(pw,b.i32_val(9)),b.shl(lane,b.i32_val(3)));
    Value coord=packetLinear(b,ordinal,p->basis),row=b.udiv(coord,b.i32_val(p->columns)),col=b.urem(coord,b.i32_val(p->columns));
    Value low=b.add(b.ptrtoint(i32_ty,smem.getBase()),b.shl(pw,b.i32_val(10)));
    Value packed=b.or_(low,b.shl(b.and_(pw,b.i32_val(structured?3:1)),b.i32_val(structured?16:18)));
    Value dst=b.inttoptr(ptr_ty(ctx,3),packed);
    if(global && structured && wholeColumn) {
      Value rowRebase=b.select(b.icmp_sge(a.getIndices()[0],b.i32_val(0)),a.getIndices()[0],b.i32_val(0));
      Value vi=b.add(row,b.sub(a.getIndices()[0],rowRebase));
      Value offset=b.add(b.shl(col,b.i32_val(1)),b.i32_val(16));
      {auto call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),vi,offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});packetPhysicalAliasScopes(op,call.getOperation(),false);}
    } else if(boundedRaw) {
      Value local=b.shl(b.add(b.mul(b.zext(i64_ty,row),b.zext(i64_ty,a.getShape()[1])),b.zext(i64_ty,col)),b.i64_val(1));
      Value relative=b.add(local,b.sub(rawUniform,rebase));
      Value offset=b.trunc(i32_ty,relative);
      if(!wholeColumn) {
        Value gc=b.add(colStart,b.zext(i64_ty,col));
        Value valid=b.and_(b.icmp_sge(gc,b.i64_val(0)),b.icmp_ule(b.add(gc,b.i64_val(8)),b.zext(i64_ty,a.getShape()[1])));
        valid=b.and_(valid,b.icmp_eq(b.and_(gc,b.i64_val(7)),b.i64_val(0)));
        offset=b.select(valid,offset,b.i32_val(0x80000000u));
      }
      {auto call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});packetPhysicalAliasScopes(op,call.getOperation(),false);}
    } else if(global) {
      Value gr=b.add(b.sext(i64_ty,a.getIndices()[0]),b.zext(i64_ty,row));
      Value gc=b.add(b.sext(i64_ty,a.getIndices()[1]),b.zext(i64_ty,col));
      Value positive=b.and_(b.icmp_sge(gr,b.i64_val(0)),b.icmp_sge(gc,b.i64_val(0)));
      Value inBounds=b.and_(b.icmp_ult(gr,b.zext(i64_ty,a.getShape()[0])),
          b.icmp_ule(b.add(gc,b.i64_val(8)),b.zext(i64_ty,a.getShape()[1])));
      Value aligned=b.icmp_eq(b.and_(gc,b.i64_val(7)),b.i64_val(0));
      Value valid=b.and_(b.and_(positive,inBounds),b.and_(aligned,b.icmp_ne(a.getIndices()[2],b.i32_val(0))));
      if(structured) {
        Value vi=b.select(valid,b.trunc(i32_ty,gr),b.i32_val(0x80000000u));
        Value offset=b.trunc(i32_ty,b.add(b.shl(gc,b.i64_val(1)),b.i64_val(16)));
        {auto call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),vi,offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});packetPhysicalAliasScopes(op,call.getOperation(),false);}
      } else {
        Value element=b.add(b.mul(gr,b.zext(i64_ty,a.getShape()[1])),gc);
        Value offset=b.select(valid,b.trunc(i32_ty,b.sub(b.shl(element,b.i64_val(1)),rebase)),b.i32_val(0x80000000u));
        {auto call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});packetPhysicalAliasScopes(op,call.getOperation(),false);}
      }
    } else {
    if(structured) {auto call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),row,b.add(b.shl(col,b.i32_val(1)),b.i32_val(16)),b.i32_val(0),b.i32_val(0),b.i32_val(0)});packetPhysicalAliasScopes(op,call.getOperation(),false);}
    else {
      Value offset=b.shl(b.add(b.mul(row,a.getShape()[1]),col),b.i32_val(1));
      Value valid=b.icmp_ult(b.add(col,a.getIndices()[1]),a.getShape()[1]);
      offset=b.select(valid,offset,b.i32_val(0x80000000u));
      {auto call=LLVM::createLLVMCallOp(rw,loc,fn,{resource,dst,b.i32_val(16),offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});packetPhysicalAliasScopes(op,call.getOperation(),false);}
    }
    }
  }
  rw.replaceOp(op,b.i32_val(0));return success();
}
static LogicalResult lowerWholePacketConsumer(LocalLoadOp op,LocalLoadOpAdaptor a,
    const LLVMTypeConverter &converter,ConversionPatternRewriter &rw) {
  auto p=packetLayout(op,op.getSrc().getType());if(!p)return op.emitError("wholepacket: invalid consumer source chart");
  auto encoding=cast<HCUMlsSharedEncodingAttr>(op.getSrc().getType().getEncoding());
  unsigned role=encoding.getOpIdx();
  auto semantics=*wp::PacketSemantics::get(encoding.getVersion(),role,encoding.getAlt2Kind());
  bool structured=semantics.transfer==wp::Role::StructuredRows;
  auto request=op->getParentOfType<ModuleOp>()->getAttrOfType<IntegerAttr>(
      "ttg.hcu.packet_matrix_read_partition");
  auto plan=mlir::triton::HCU::planWholePacketConsumer(op.getType(),semantics,*p,
      role,encoding.getVersion(),request && request.getInt()==1);
  if(!plan)return op.emitError("wholepacket: invalid or unbounded consumer footprint");
  auto &rewriter=rw;
  auto loc=op.getLoc();auto b=TritonLLVMOpBuilder(loc,rw);auto *ctx=rw.getContext();
  auto elem=converter.convertType(op.getType().getElementType());auto smem=getSharedMemoryObjectFromStruct(loc,a.getSrc(),elem,rw);
  auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),warp=StringAttr::get(ctx,"warp");
  auto logical=triton::gpu::toLinearLayout(op.getType());logical=logical.sublayout({reg,lane,warp},llvm::to_vector(logical.getOutDimNames()));
  auto [laneId,warpId]=getLaneAndWarpId(rw,loc);
  unsigned registers=logical.getInDimSize(reg);
  SmallVector<Value> values(registers);
  bool matrix=plan->matrix,matrixAlt=plan->matrixAlt,repartition=plan->repartition;
  unsigned width=plan->width;
  const auto &registerAxes=plan->registerAxes;
  const auto &matrixByAnchor=plan->matrixByAnchor;
  const auto &groups=plan->groups;
  const auto &representatives=plan->representatives;
  const auto &deltas=plan->deltas;
  const auto &unique=plan->unique;
  LLVM::LLVMFuncOp matrixFn;
  if(matrix)matrixFn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rw,
      matrixAlt?"llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3":
          "llvm.hcu.ds.read.m32x16.b16.v4i32.p3",vec_ty(i32_ty,4),{ptr_ty(ctx,3),i32_ty});
  llvm::DenseMap<unsigned,Value> pointers;
  for(unsigned r:unique) {
    Value regId=b.i32_val(r),readLane=laneId;
    if(matrix) {
      Value sourceReg=b.and_(b.lshr(laneId,b.i32_val(2)),b.i32_val(3));
      if(!matrixAlt)sourceReg=b.or_(sourceReg,
          b.shl(b.and_(b.lshr(laneId,b.i32_val(1)),b.i32_val(1)),b.i32_val(2)));
      if(repartition) {
        regId=b.i32_val(r);
        for(unsigned bit=0;bit<3;++bit) {
          Value digit=b.and_(b.lshr(sourceReg,b.i32_val(bit)),b.i32_val(1));
          regId=b.or_(regId,b.shl(digit,b.i32_val(registerAxes[bit])));
        }
      } else regId=b.or_(sourceReg,b.i32_val(r));
      readLane=b.or_(b.and_(laneId,b.i32_val(48)),
          b.shl(b.and_(laneId,b.i32_val(matrixAlt?3:1)),b.i32_val(matrixAlt?2:3)));
    }
    auto xy=applyLinearLayout(loc,rw,logical,{{reg,regId},{lane,readLane},{warp,warpId}});
    Value row=semantics.transpose?xy[1].second:xy[0].second;
    Value col=semantics.transpose?xy[0].second:xy[1].second;
    Value coord=b.add(b.mul(row,b.i32_val(p->columns)),col);
    Value half=packetPhysical(b,coord,*p,structured);
    pointers[r]=b.gep(ptr_ty(ctx,3),elem,smem.getBase(),half);
  }
  for(unsigned groupIndex=0;groupIndex<registers/width;++groupIndex) {
    unsigned r=repartition?groups[groupIndex]:groupIndex*width;
    Value ptr=pointers.lookup(representatives[r]);unsigned byteImmediate=2*deltas[r];
    if(!matrix && deltas[r])ptr=b.gep(ptr_ty(ctx,3),elem,ptr,b.i32_val(deltas[r]));
    if(matrix) {
      auto call=LLVM::createLLVMCallOp(rw,loc,matrixFn,{ptr,b.i32_val(byteImmediate)});
      packetPhysicalAliasScopes(op,call.getOperation(),true);
      call->setAttr("hcu.packet.reader",rw.getStringAttr(matrixAlt?"matrix-alt-finite-proof":"matrix-finite-proof"));
      Value result=b.bitcast(call.getResult(),vec_ty(elem,8));
      for(unsigned j=0;j<8;++j)
        values[repartition?matrixByAnchor[r][j]:r+j]=b.extract_element(elem,result,b.i32_val(j));
    } else if(width>1) {
      auto read=LLVM::LoadOp::create(rw,loc,vec_ty(elem,width),ptr,2*width,false,false);
      packetPhysicalAliasScopes(op,read.getOperation(),true);
      read->setAttr("hcu.packet.reader",rw.getStringAttr("contiguous-finite-proof"));
      for(unsigned j=0;j<width;++j)values[r+j]=b.extract_element(elem,read.getResult(),b.i32_val(j));
    } else {
      auto read=LLVM::LoadOp::create(rw,loc,elem,ptr,2,false,false);
      packetPhysicalAliasScopes(op,read.getOperation(),true);
      read->setAttr("hcu.packet.reader",rw.getStringAttr("scalar-fallback"));
      values[r]=read.getResult();
    }
  }
  rw.replaceOp(op,packLLElements(loc,&converter,values,rw,op.getType()));return success();
}
} // namespace
