#include "Dialect/TritonHCUGPU/IR/RawWindowLayout.h"
namespace {
namespace rw=mlir::triton::hcugpu::rawwindow;
namespace rh=mlir::triton::hcugpu;
static Value rawCapture(Value v) {
  auto arg=dyn_cast<BlockArgument>(v);if(!arg)return v;
  auto parts=dyn_cast<WarpSpecializePartitionsOp>(arg.getOwner()->getParentOp());
  if(!parts||arg.getOwner()!=&arg.getOwner()->getParent()->front())return v;
  auto captures=parts.getParentOp().getExplicitCaptures();
  return arg.getArgNumber()<captures.size()?rawCapture(captures[arg.getArgNumber()]):v;
}
static bool rawUniform(Value v,llvm::SmallPtrSetImpl<void*>& seen) {
  if(!v.getType().isIntOrIndex())return false;
  Value capture=rawCapture(v);if(capture!=v)return rawUniform(capture,seen);
  if(!seen.insert(v.getAsOpaquePointer()).second)return true;
  if(auto arg=dyn_cast<BlockArgument>(v)) {
    Block* block=arg.getOwner();unsigned n=arg.getArgNumber();
    if(auto f=dyn_cast<triton::FuncOp>(block->getParentOp()))
      if(block==&f.getBody().front())return true;
    if(block->hasNoPredecessors())return false;
    for(Block* pred:block->getPredecessors()) {
      auto term=pred->getTerminator();bool edge=false;
      if(auto br=dyn_cast<cf::BranchOp>(term)) {
        if(br.getDest()!=block||!rawUniform(br.getDestOperands()[n],seen))return false;edge=true;
      } else if(auto br=dyn_cast<cf::CondBranchOp>(term)) {
        if(!rawUniform(br.getCondition(),seen))return false;
        if(br.getTrueDest()==block){if(!rawUniform(br.getTrueDestOperands()[n],seen))return false;edge=true;}
        if(br.getFalseDest()==block){if(!rawUniform(br.getFalseDestOperands()[n],seen))return false;edge=true;}
      }
      if(!edge)return false;
    }return true;
  }
  auto op=v.getDefiningOp();if(!op)return false;
  if(isa<triton::GetProgramIdOp>(op))return true;
  if(!op->getName().getStringRef().starts_with("arith.")||op->getNumRegions())return false;
  for(Value operand:op->getOperands())if(!rawUniform(operand,seen))return false;
  return true;
}
static bool rawUniform(Value v){llvm::SmallPtrSet<void*,32> seen;return rawUniform(v,seen);}
static bool rawOwner(Value value) {
  value=rawCapture(value);
  auto ty=dyn_cast<MemDescType>(value.getType());
  auto index=value.getDefiningOp<MemDescIndexOp>();
  auto owner=index?rawCapture(index.getSrc()).getDefiningOp<LocalAllocOp>():LocalAllocOp{};
  if(!ty||ty.getShape()!=ArrayRef<int64_t>{rw::Elements}||
     !(ty.getElementType().isF16()||ty.getElementType().isBF16())||!owner||owner.getSrc()||
     owner.getType().getRank()!=2||owner.getType().getShape()[1]!=rw::Elements||
     owner.getType().getShape()[0]<1||owner.getType().getShape()[0]>2||
     owner.getAlignmentOrDefault()<rw::Bytes)return false;
  auto enc=dyn_cast<SwizzledSharedEncodingAttr>(owner.getType().getEncoding());
  if(!enc||enc.getVec()!=1||enc.getPerPhase()!=1||enc.getMaxPhase()!=1||
      enc.getOrder()!=ArrayRef<unsigned>{0}||ty.getEncoding()!=owner.getType().getEncoding())return false;
  auto c=getSignedI64Constant(index.getIndex());
  return c&&*c>=0&&*c<owner.getType().getShape()[0];
}
static std::optional<rw::SourceLayout> rawChart(ArrayRef<int32_t> shape,ArrayRef<int32_t> basis) {
  if(shape.size()!=2||shape[0]<=0||shape[1]<=0||basis.size()!=14)return {};
  std::array<unsigned,14> bs{};for(unsigned i=0;i<14;++i)bs[i]=basis[i];
  return rw::SourceLayout::get(shape[0],shape[1],bs);
}
static bool rawProducerDomain(Operation* op) {
  if(lookupNumWarps(op)!=4)return false;
  if(auto parts=op->getParentOfType<WarpSpecializePartitionsOp>()) {
    auto ws=parts.getParentOp();Region* region=op->getParentRegion();
    while(region&&region->getParentOp()!=parts)region=region->getParentOp()->getParentRegion();
    if(!region)return false;
    unsigned i=region->getRegionNumber();auto starts=ws.getWarpGroupStartIds();
    if(i>=ws.getPartitionNumWarps().size()||ws.getPartitionNumWarps()[i]!=4||
       !starts||i>=starts->size()||(*starts)[i]%4)return false;
  }
  // Scalar loop/branch conditions retain full-wave issue. Partition dispatch
  // itself is checked above and is not mistaken for lane divergence.
  Region* region=op->getParentRegion();
  bool uniform=true;region->walk([&](cf::CondBranchOp br){uniform&=rawUniform(br.getCondition());});
  return uniform;
}
struct RawReadPlan {rw::SourceLayout chart;unsigned width;};
static std::optional<RawReadPlan> rawReadPlan(rh::RawWindowReadOp op,bool checkOwner=true) {
  auto chart=rawChart(op.getStorageShape(),op.getBasis());auto ty=cast<RankedTensorType>(op.getType());
  for(StringRef name:{"hcu.full_read_completion","hcu.global_read_completion"})
    if(auto attr=op->getAttr(name);attr&&!isa<UnitAttr>(attr))return {};
  auto origin=op.getOrigin(),order=op.getOrder();
  if(!chart||(checkOwner&&!rawOwner(op.getSource()))||ty.getRank()!=2||
     ty.getElementType()!=op.getSource().getType().getElementType()||origin.size()!=2||
     order.size()!=2||order[0]<0||order[0]>1||order[1]!=1-order[0]||origin[0]<0||origin[1]<0||
     int64_t(origin[0])+ty.getShape()[order[0]]>chart->rows||
     int64_t(origin[1])+ty.getShape()[order[1]]>chart->columns)return {};
  auto ll=triton::gpu::toLinearLayout(ty);auto*ctx=op.getContext();
  auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),warp=StringAttr::get(ctx,"warp"),block=StringAttr::get(ctx,"block");
  if(!ll.hasInDim(reg)||!ll.hasInDim(lane)||!ll.hasInDim(warp)||!ll.hasInDim(block)||
      ll.getInDimSize(lane)!=64||ll.getInDimSize(block)!=1)return {};
  auto address=[&](unsigned r,unsigned l,unsigned w) {
    auto xy=ll.apply({{reg,r},{lane,l},{warp,w},{block,0}});
    unsigned row=xy[order[0]].second+origin[0],col=xy[order[1]].second+origin[1];
    return chart->address(row*chart->columns+col);
  };
  unsigned registers=ll.getInDimSize(reg),waves=ll.getInDimSize(warp);
  for(unsigned width:{8u,4u,2u,1u}) {
    if(registers%width)continue;bool ok=true;
    for(unsigned w=0;w<waves&&ok;++w)for(unsigned l=0;l<64&&ok;++l)
      for(unsigned r=0;r<registers&&ok;r+=width) {
        unsigned base=address(r,l,w);if(base%width){ok=false;break;}
        for(unsigned j=1;j<width;++j)if(address(r+j,l,w)!=base+j){ok=false;break;}
      }
    if(ok)return RawReadPlan{*chart,width};
  }return {};
}
static LogicalResult verifyRawWindows(ModuleOp mod,StringRef arch) {
  bool valid=true;
  mod.walk([&](Operation* op){
    if(!isa<rh::RawWindowCopyOp,rh::RawWindowReadOp,rh::RawWindowWaitOp,rh::WarpGroupIdOp>(op))return;
    auto reject=[&](StringRef msg){op->emitError(msg);valid=false;};
    if(arch!="gfx936"){reject("raw32KiB instruction ABI requires gfx936");return;}
    if(auto group=dyn_cast<rh::WarpGroupIdOp>(op)) {
      int64_t size=group.getGroupSize();
      if(size<=0||!llvm::isPowerOf2_64(size)||size>lookupNumWarps(op))
        reject("warp group size must be a positive power of two within this partition");
    } else if(auto copy=dyn_cast<rh::RawWindowCopyOp>(op)) {
      auto chart=rawChart(copy.getStorageShape(),copy.getBasis());
      auto ptr=dyn_cast<triton::PointerType>(copy.getBase().getType());
      int64_t pitch=copy.getPitch();
      if(!chart||!rawOwner(copy.getDest())||!ptr||ptr.getAddressSpace()!=1||
         ptr.getPointeeType()!=copy.getDest().getType().getElementType()||
         !structuredRegisterBaseUniform(rawCapture(copy.getBase()))||structuredRegisterAlignment(rawCapture(copy.getBase()))<16||
         pitch<chart->columns||pitch%8||uint64_t(chart->rows)*pitch*2+16>0x7fffffffull||
         !rawUniform(copy.getRows())||!rawUniform(copy.getRowOrigin())||
         !rawUniform(copy.getColumnOrigin())||!rawUniform(copy.getEnabled())||
         !packetMultiple(copy.getColumnOrigin(),8)||!rawProducerDomain(copy))
        reject("raw window copy requires a complete aligned source chart, bounded row-major pitch, uniform coordinates, and exactly four producer waves");
      if(!copy.getToken().hasOneUse()||!isa<rh::RawWindowWaitOp>(*copy.getToken().getUsers().begin()))
        reject("raw window copy must have one explicit matching VM wait");
    } else if(auto read=dyn_cast<rh::RawWindowReadOp>(op)) {
      if(op->hasAttr("hcu.full_read_completion") && !isa<UnitAttr>(op->getAttr("hcu.full_read_completion")))
        reject("raw read completion request must be a unit attribute");
      if(!rawReadPlan(read))reject("raw window read has invalid full owner, source chart, or distributed subtile");
    } else if(auto wait=dyn_cast<rh::RawWindowWaitOp>(op)) {
      if(!wait.getToken().getDefiningOp<rh::RawWindowCopyOp>())reject("raw window wait requires its direct copy token");
    }
  });return success(valid);
}
static Value rawApply(TritonLLVMOpBuilder& b,const std::array<unsigned,14>& basis,Value v) {
  Value result=b.i32_val(0);
  for(unsigned i=0;i<14;++i)if(basis[i]) {
    Value bit=b.and_(b.lshr(v,b.i32_val(i)),b.i32_val(1));
    result=b.xor_(result,b.mul(bit,b.i32_val(basis[i])));
  }return result;
}
struct RawWindowCopyConversion : ConvertOpToLLVMPattern<rh::RawWindowCopyOp> {
  using ConvertOpToLLVMPattern<rh::RawWindowCopyOp>::ConvertOpToLLVMPattern;
  // Compilation-local cache. Relative warp id is uniform within each wave,
  // invariant across loop iterations, and anchored within its WS partition.
  // Moving this exact thread-id-derived query within that partition is safe
  // even when later control changes the active mask: every active lane agrees.
  mutable llvm::DenseMap<Region*,Value> scalarWaves;
  Value scalarWave(rh::RawWindowCopyOp op,ConversionPatternRewriter& rewriter) const {
    Region* region=op->getParentRegion();
    while(region && !isa<WarpSpecializeOp,WarpSpecializePartitionsOp,triton::FuncOp,LLVM::LLVMFuncOp>(region->getParentOp()))
      region=region->getParentOp()->getParentRegion();
    assert(region && !region->empty() && "raw copy requires a callable/WS region");
    auto found=scalarWaves.find(region);if(found!=scalarWaves.end())return found->second;
    OpBuilder::InsertionGuard guard(rewriter);rewriter.setInsertionPointToStart(&region->front());
    auto ids=getLaneAndWarpId(rewriter,op.getLoc());
    Value wave=rewriter.create<ROCDL::ReadfirstlaneOp>(op.getLoc(),i32_ty,ids.second);
    scalarWaves[region]=wave;return wave;
  }
  LogicalResult matchAndRewrite(rh::RawWindowCopyOp op,OpAdaptor a,ConversionPatternRewriter& rewriter) const override {
    auto chart=rawChart(op.getStorageShape(),op.getBasis());if(!chart)return failure();
    auto loc=op.getLoc();auto b=TritonLLVMOpBuilder(loc,rewriter);auto*ctx=rewriter.getContext();
    Type elem=getTypeConverter()->convertType(op.getDest().getType().getElementType());
    auto smem=getSharedMemoryObjectFromStruct(loc,a.getDest(),elem,rewriter);
    auto [lane,wave0]=getLaneAndWarpId(rewriter,loc);
    (void)wave0;
    Value wave=scalarWave(op,rewriter);
    Value rows=b.select(b.icmp_sgt(a.getRows(),b.i32_val(0)),a.getRows(),b.i32_val(0));
    Value rowBase=b.select(b.icmp_sgt(a.getRowOrigin(),b.i32_val(0)),a.getRowOrigin(),b.i32_val(0));
    rowBase=b.select(b.icmp_slt(rowBase,rows),rowBase,rows);
    Value count=b.sub(rows,rowBase);count=b.select(b.icmp_ult(count,b.i32_val(chart->rows)),count,b.i32_val(chart->rows));
    // Complete K windows only. Invalid/disabled windows issue zero-fill reads
    // with NR0; the address is still based on a clamped source allocation row.
    Value kValid=b.icmp_ule(a.getColumnOrigin(),b.i32_val(op.getPitch()-chart->columns));
    Value enabled=b.and_(a.getEnabled(),kValid);
    Value kBase=b.select(kValid,a.getColumnOrigin(),b.i32_val(0));
    Value baseIndex=b.add(b.mul(b.zext(i64_ty,rowBase),b.i64_val(op.getPitch())),b.zext(i64_ty,kBase));
    Value base64=b.sub(b.add(b.ptrtoint(i64_ty,a.getBase()),b.shl(baseIndex,b.i64_val(1))),b.i64_val(16));
    Value nr=b.add(b.shl(b.mul(count,b.i32_val(op.getPitch())),b.i32_val(1)),b.i32_val(16));
    nr=b.select(enabled,nr,b.i32_val(0));
    Value resource=b.undef(vec_ty(i32_ty,4));
    SmallVector<Value> words{b.trunc(i32_ty,base64),b.and_(b.trunc(i32_ty,b.lshr(base64,b.i64_val(32))),b.i32_val(0xffff)),nr,b.i32_val(0x20000)};
    for(unsigned i=0;i<4;++i)resource=b.insert_element(vec_ty(i32_ty,4),resource,words[i],b.i32_val(i));
    auto fn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rewriter,"llvm.hcu.raw.buffer.load.lds",
      LLVM::LLVMVoidType::get(ctx),{vec_ty(i32_ty,4),ptr_ty(ctx,3),i32_ty,i32_ty,i32_ty,i32_ty,i32_ty});
    Value ordinal=b.or_(b.shl(wave,b.i32_val(9)),b.shl(lane,b.i32_val(3)));
    Value rawBase=b.add(b.ptrtoint(i32_ty,smem.getBase()),b.shl(wave,b.i32_val(10)));
    rawBase=b.or_(rawBase,b.shl(wave,b.i32_val(17)));
    for(unsigned copy=0;copy<8;++copy) {
      Value logical=rawApply(b,chart->basis,b.or_(ordinal,b.i32_val(copy*2048)));
      Value n=b.lshr(logical,b.i32_val(llvm::Log2_32(chart->columns)));
      Value k=b.and_(logical,b.i32_val(chart->columns-1));
      Value relativeRow=b.add(n,b.sub(a.getRowOrigin(),rowBase));
      Value offset=b.add(b.shl(b.add(b.mul(relativeRow,b.i32_val(op.getPitch())),k),b.i32_val(1)),b.i32_val(16));
      offset=b.select(b.icmp_ult(relativeRow,count),offset,b.i32_val(0xfffffff0u));
      Value packedInteger=b.add(rawBase,b.i32_val(copy*4096));
      Value packed=b.inttoptr(ptr_ty(ctx,3),packedInteger);
      LLVM::createLLVMCallOp(rewriter,loc,fn,{resource,packed,b.i32_val(16),offset,b.i32_val(0),b.i32_val(0),b.i32_val(0)});
    }
    rewriter.replaceOp(op,b.i32_val(0));return success();
  }
};
struct RawWindowReadConversion : ConvertOpToLLVMPattern<rh::RawWindowReadOp> {
  using ConvertOpToLLVMPattern<rh::RawWindowReadOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(rh::RawWindowReadOp op,OpAdaptor a,ConversionPatternRewriter& rewriter) const override {
    auto plan=rawReadPlan(op,false);if(!plan)return failure();auto loc=op.getLoc();auto b=TritonLLVMOpBuilder(loc,rewriter);
    auto ty=cast<RankedTensorType>(op.getType());Type elem=getTypeConverter()->convertType(ty.getElementType());
    auto smem=getSharedMemoryObjectFromStruct(loc,a.getSource(),elem,rewriter);auto*ctx=rewriter.getContext();
    auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),warp=StringAttr::get(ctx,"warp");
    // This is a local scheduling completion, never shared-DMA publication.
    // A vector prefetch from the preceding K tile is complete before this read.
    if(op->hasAttr("hcu.global_read_completion")) {
      auto fn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rewriter,
        "llvm.amdgcn.s.waitcnt",LLVM::LLVMVoidType::get(ctx),{i32_ty});
      LLVM::createLLVMCallOp(rewriter,loc,fn,{b.i32_val(0x0f70)});
    }
    auto ll=triton::gpu::toLinearLayout(ty);unsigned registers=ll.getInDimSize(reg);
    ll=ll.sublayout({reg,lane,warp},llvm::to_vector(ll.getOutDimNames()));
    auto [laneId,warpId]=getLaneAndWarpId(rewriter,loc);SmallVector<Value> values;
    for(unsigned r=0;r<registers;r+=plan->width) {
      auto xy=applyLinearLayout(loc,rewriter,ll,{{reg,b.i32_val(r)},{lane,laneId},{warp,warpId}});
      Value row=b.add(xy[op.getOrder()[0]].second,b.i32_val(op.getOrigin()[0]));
      Value col=b.add(xy[op.getOrder()[1]].second,b.i32_val(op.getOrigin()[1]));
      Value ordinal=rawApply(b,plan->chart.inverseBasis,b.add(b.mul(row,b.i32_val(plan->chart.columns)),col));
      Value wave=b.and_(b.lshr(ordinal,b.i32_val(9)),b.i32_val(3));
      Value physical=b.or_(b.and_(ordinal,b.i32_val(0x3e00)),b.and_(b.add(ordinal,b.shl(wave,b.i32_val(4))),b.i32_val(511)));
      Value ptr=b.gep(ptr_ty(ctx,3),elem,smem.getBase(),physical);
      Type readType=vec_ty(elem,plan->width);
      Value loaded=rewriter.create<LLVM::LoadOp>(loc,readType,ptr,plan->width*2);
      for(unsigned j=0;j<plan->width;++j)values.push_back(b.extract_element(elem,loaded,b.i32_val(j)));
    }
    if(op->hasAttr("hcu.full_read_completion")) {
      auto fn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rewriter,
        "llvm.amdgcn.s.waitcnt",LLVM::LLVMVoidType::get(ctx),{i32_ty});
      LLVM::createLLVMCallOp(rewriter,loc,fn,{b.i32_val(0xc07f)});
    }
    rewriter.replaceOp(op,packLLElements(loc,getTypeConverter(),values,rewriter,ty));return success();
  }
};
struct RawWarpGroupIdConversion : ConvertOpToLLVMPattern<rh::WarpGroupIdOp> {
  using ConvertOpToLLVMPattern<rh::WarpGroupIdOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(rh::WarpGroupIdOp op,OpAdaptor,ConversionPatternRewriter& rewriter) const override {
    auto loc=op.getLoc();auto b=TritonLLVMOpBuilder(loc,rewriter);
    auto ids=getLaneAndWarpId(rewriter,loc);
    Value wave=rewriter.create<ROCDL::ReadfirstlaneOp>(loc,i32_ty,ids.second);
    rewriter.replaceOp(op,b.udiv(wave,b.i32_val(op.getGroupSize())));return success();
  }
};
struct RawWindowWaitConversion : ConvertOpToLLVMPattern<rh::RawWindowWaitOp> {
  using ConvertOpToLLVMPattern<rh::RawWindowWaitOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(rh::RawWindowWaitOp op,OpAdaptor,ConversionPatternRewriter& rewriter) const override {
    auto loc=op.getLoc();auto b=TritonLLVMOpBuilder(loc,rewriter);
    auto fn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rewriter,"llvm.amdgcn.s.waitcnt",
      LLVM::LLVMVoidType::get(rewriter.getContext()),{i32_ty});
    // VM0, retain maximal EXP/LGKM fields. Source kernel owns CTA ordering.
    LLVM::createLLVMCallOp(rewriter,loc,fn,{b.i32_val(0x0f70)});
    rewriter.eraseOp(op);return success();
  }
};
}
