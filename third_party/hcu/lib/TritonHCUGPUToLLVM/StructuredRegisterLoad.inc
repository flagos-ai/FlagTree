// Explicit rank-two GPR loads. This path is separate from all MLS/LDS gates.
// Request attributes are never proof: preflight discards imported witnesses,
// verifies actual SSA/layout, then lowering consumes a same-pass witness.
namespace {
constexpr llvm::StringLiteral StructuredRegisterRequest="hcu.structured_register";
constexpr llvm::StringLiteral StructuredRegisterWidth="hcu.structured_register.internal_width";
constexpr llvm::StringLiteral StructuredRegisterOrder="hcu.structured_register.internal_order";

// Register numbering is not an address order. A pure split/transpose may put
// a distant column bit before the contiguous bits without changing ownership.
// Group complete vectors by their actual LinearLayout coordinates, then prove
// the same grouping for every lane and wave. The result is only a permutation
// of this thread's registers; it never authorizes a cross-thread exchange.
static SmallVector<int32_t> structuredRegisterOrder(
    RankedTensorType type, unsigned width) {
  auto *ctx=type.getContext();
  auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),
       warp=StringAttr::get(ctx,"warp"),block=StringAttr::get(ctx,"block");
  auto ll=triton::gpu::toLinearLayout(type);
  unsigned count=ll.getInDimSize(reg);
  if(count%width)return {};
  SmallVector<int32_t> order;
  for(unsigned r=0;r<count;++r)order.push_back(r);
  auto at=[&](unsigned r,unsigned l,unsigned w) {
    return ll.apply({{reg,int32_t(r)},{lane,int32_t(l)},
                     {warp,int32_t(w)},{block,0}});
  };
  auto valid=[&]() {
    for(unsigned i=0;i<count;i+=width)
      for(unsigned w=0;w<ll.getInDimSize(warp);++w)
        for(unsigned l=0;l<ll.getInDimSize(lane);++l) {
          auto xy=at(order[i],l,w);
          if(xy[1].second%width || xy[1].second+width>type.getShape()[1])
            return false;
          for(unsigned j=1;j<width;++j) {
            auto z=at(order[i+j],l,w);
            if(z[0].second!=xy[0].second || z[1].second!=xy[1].second+j)
              return false;
          }
        }
    return true;
  };
  // Keep instruction ordering exactly unchanged for already legal vectors.
  if(valid())return order;
  SmallVector<std::pair<int32_t,int32_t>> coordinates;
  for(unsigned r=0;r<count;++r) {
    auto xy=at(r,0,0);
    coordinates.emplace_back(xy[0].second,xy[1].second);
  }
  llvm::stable_sort(order,[&](unsigned a,unsigned b) {
    return coordinates[a]<coordinates[b];
  });
  return valid()?order:SmallVector<int32_t>{};
}
// Descriptor DATA must be CTA-uniform, independent of unrelated execution
// branches. In particular warp_group_id is never a valid descriptor operand.
static bool structuredRegisterDataUniform(Value v) {
  llvm::SmallPtrSet<void *,32> seen;
  return mlir::triton::HCU::memory_analysis::uniformValue(v,seen);
}
// EXECUTION CONTROL may differ between waves but must agree across all lanes
// of each wave. Keep this local to explicit structured-register loads; neither
// the ordinary descriptor-uniformity proof acquires no new authority.
static bool structuredRegisterWaveControl(Value v,llvm::SmallPtrSetImpl<void *>& seen) {
  if(!v.getType().isIntOrIndex())return false;
  if(!seen.insert(v.getAsOpaquePointer()).second)return true;
  if(auto arg=dyn_cast<BlockArgument>(v)) {
    Block* dest=arg.getOwner();unsigned n=arg.getArgNumber();
    if(auto f=dyn_cast<triton::FuncOp>(dest->getParentOp()))
      if(dest==&f.getBody().front())return true;
    if(dest->hasNoPredecessors())return false;
    for(Block* pred:dest->getPredecessors()) {
      Operation* term=pred->getTerminator();bool edge=false;
      if(auto br=dyn_cast<cf::BranchOp>(term)) {
        if(br.getDest()!=dest||n>=br.getDestOperands().size()||!structuredRegisterWaveControl(br.getDestOperands()[n],seen))return false;
        edge=true;
      } else if(auto br=dyn_cast<cf::CondBranchOp>(term)) {
        if(!structuredRegisterWaveControl(br.getCondition(),seen))return false;
        if(br.getTrueDest()==dest) {
          if(n>=br.getTrueDestOperands().size()||!structuredRegisterWaveControl(br.getTrueDestOperands()[n],seen))return false;
          edge=true;
        }
        if(br.getFalseDest()==dest) {
          if(n>=br.getFalseDestOperands().size()||!structuredRegisterWaveControl(br.getFalseDestOperands()[n],seen))return false;
          edge=true;
        }
      }
      if(!edge)return false;
    }return true;
  }
  Operation* op=v.getDefiningOp();if(!op)return false;
  if(op->getName().getStringRef()=="hcug.warp_group_id") {
    auto size=op->getAttrOfType<IntegerAttr>("groupSize");
    if(op->getNumOperands()||op->getNumResults()!=1||!v.getType().isInteger(32)||!size||!size.getType().isInteger(32))return false;
    int64_t count=size.getInt();
    return count>0&&!(uint64_t(count)&uint64_t(count-1))&&count<=triton::gpu::lookupNumWarps(op);
  }
  if(isa<triton::GetProgramIdOp>(op))return true;
  if(!op->getName().getStringRef().starts_with("arith.")||op->getNumRegions())return false;
  for(Value operand:op->getOperands())if(!structuredRegisterWaveControl(operand,seen))return false;
  return true;
}
static bool structuredRegisterExecutionUniform(Operation* site) {
  auto f=site->getParentOfType<triton::FuncOp>();if(!f)return false;
  bool result=true;
  f.walk([&](Operation* op) {
    if(auto br=dyn_cast<cf::CondBranchOp>(op)) {
      llvm::SmallPtrSet<void *,32> seen;
      result&=structuredRegisterWaveControl(br.getCondition(),seen);
    } else if(op->getNumSuccessors()>1) {
      // No switch/indirect/unknown conditional-control model is inferred.
      result=false;
    }
  });return result;
}
static bool structuredRegisterBaseUniform(Value v) {
  v = mlir::triton::HCU::resolveWarpSpecializeCapture(v);
  if(auto arg=dyn_cast<BlockArgument>(v))
    if(auto f=dyn_cast<triton::FuncOp>(arg.getOwner()->getParentOp()))
      return arg.getOwner()==&f.getBody().front();
  if(auto add=v.getDefiningOp<triton::AddPtrOp>())
    return structuredRegisterBaseUniform(add.getPtr()) && structuredRegisterDataUniform(add.getOffset());
  return false;
}
static unsigned structuredRegisterAlignment(Value v) {
  v = mlir::triton::HCU::resolveWarpSpecializeCapture(v);
  if(auto arg=dyn_cast<BlockArgument>(v))
    if(auto f=dyn_cast<triton::FuncOp>(arg.getOwner()->getParentOp()))
      if(arg.getOwner()==&f.getBody().front())
        if(auto a=f.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),"tt.divisibility"))
          if(a.getInt()>0)return unsigned(a.getInt());
  return 2; // f16/bf16 element pointer; otherwise retain scalar16-bit reads.
}
static LogicalResult verifyStructuredRegisterLoads(ModuleOp mod,StringRef arch) {
  mod.walk([&](Operation *op){
    op->removeAttr(StructuredRegisterWidth);
    op->removeAttr(StructuredRegisterOrder);
  });
  if(auto a=mod->getAttr("ttg.hcu.structured_register_prepack_address");a&&!isa<BoolAttr>(a))
    return mod.emitError("structured-register: prepack-address option must be BoolAttr");
  if(auto a=mod->getAttr("ttg.hcu.structured_register_unclamped_column");a&&!isa<BoolAttr>(a))
    return mod.emitError("structured-register: unclamped-column option must be BoolAttr");
  LogicalResult answer=success();
  mod.walk([&](triton::MatrixLoadOp op) {
    if(!op->hasAttr(StructuredRegisterRequest) || failed(answer))return;
    auto fail=[&](StringRef why){answer=op.emitError("structured-register: ")<<why;};
    if(!isa<UnitAttr>(op->getAttr(StructuredRegisterRequest)) || arch!="gfx936")
      return fail("requires explicit unit request on gfx936");
    auto t=dyn_cast<RankedTensorType>(op.getType());
    auto pointer=dyn_cast<triton::PointerType>(op.getBase().getType());
    if(!t || t.getRank()!=2 || !(t.getElementType().isF16() || t.getElementType().isBF16()) ||
       !isa<DistributedEncodingTrait>(t.getEncoding()) || !pointer ||
       pointer.getAddressSpace()!=1 || pointer.getPointeeType()!=t.getElementType() ||
       !structuredRegisterBaseUniform(op.getBase()))
      return fail("requires uniform AS1 base and rank-two distributed f16/bf16 result");
    if(op.getShape().size()!=2 || op.getStrides().size()!=2 || op.getIndices().size()!=3 ||
       !llvm::equal(op.getTensorShape(),t.getShape()) || !op.getBoundaryCheck().empty() ||
       op.getCache()!=triton::CacheModifier::NONE || op.getEvict()!=triton::EvictionPolicy::NORMAL || op.getIsVolatile())
      return fail("unsupported operand, tile, cache or volatile semantics");
    if(!structuredRegisterExecutionUniform(op))return fail("execution control must be scalar or validated typed wave-group uniform");
    for(Value v:op.getShape())if(!structuredRegisterDataUniform(v))return fail("shape must be CTA-uniform");
    for(Value v:op.getStrides())if(!structuredRegisterDataUniform(v))return fail("strides must be CTA-uniform");
    for(Value v:op.getIndices())if(!structuredRegisterDataUniform(v))return fail("origins/enable must be CTA-uniform");
    auto K=getSignedI64Constant(op.getShape()[1]);
    if(!K || *K<=0 || 2*uint64_t(*K)>=0x4000 ||
       getSignedI64Constant(op.getStrides()[0])!=K || getSignedI64Constant(op.getStrides()[1])!=1)
      return fail("requires proved positive K, rowstride=K, colstride=1 and 2*K<0x4000");
    if(t.getShape()[0]<1 || t.getShape()[0]>INT32_MAX || t.getShape()[1]<1 || t.getShape()[1]>*K)
      return fail("requires positive i32 tile and complete K window");
    auto *ctx=mod.getContext();auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),warp=StringAttr::get(ctx,"warp"),block=StringAttr::get(ctx,"block");
    auto ll=triton::gpu::toLinearLayout(t);
    auto nw=mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
    if(!ll.hasInDim(reg) || !ll.hasInDim(lane) || !ll.hasInDim(warp) || !ll.hasInDim(block) ||
       ll.getInDimSize(block)!=1 || ll.getInDimSize(lane)!=64 || !nw ||
       ll.getInDimSize(warp)!=nw.getInt() ||
       uint64_t(ll.getInDimSize(reg))*64*ll.getInDimSize(warp)>1048576)
      return fail("unsupported CTA/domain or finite-proof budget");
    auto at=[&](unsigned r,unsigned l,unsigned w){return ll.apply({{reg,int32_t(r)},{lane,int32_t(l)},{warp,int32_t(w)},{block,0}});};
    for(unsigned r=0;r<ll.getInDimSize(reg);++r)for(unsigned w=0;w<ll.getInDimSize(warp);++w)for(unsigned l=0;l<64;++l) {
      auto xy=at(r,l,w);
      if(xy.size()!=2 || xy[0].second<0 || xy[0].second>=t.getShape()[0] || xy[1].second<0 || xy[1].second>=t.getShape()[1])
        return fail("layout contains a coordinate outside the logical tile");
    }
    SmallVector<int32_t> order;
    auto legal=[&](unsigned width) {
      if(ll.getInDimSize(reg)%width || *K%width || structuredRegisterAlignment(op.getBase())<2*width)return false;
      llvm::SmallPtrSet<void *,32> seen;
      if(!mlir::triton::HCU::memory_analysis::multipleOf(op.getIndices()[1],width,seen))return false;
      order=structuredRegisterOrder(t,width);
      return !order.empty();
    };
    unsigned width=legal(8)?8:legal(4)?4:legal(2)?2:1;
    op->setAttr(StructuredRegisterWidth,IntegerAttr::get(IntegerType::get(ctx,32),width));
    if(width>1 && llvm::any_of(llvm::enumerate(order),[](auto entry) {
         return entry.index()!=unsigned(entry.value());
       }))
      op->setAttr(StructuredRegisterOrder,DenseI32ArrayAttr::get(ctx,order));
  });
  return answer;
}
struct StructuredRegisterLoadConversion : public ConvertOpToLLVMPattern<triton::MatrixLoadOp> {
  StructuredRegisterLoadConversion(LLVMTypeConverter &c,const HCU::TargetInfo &,ModuleAxisInfoAnalysis &,PatternBenefit benefit)
    :ConvertOpToLLVMPattern<triton::MatrixLoadOp>(c,benefit){}
  LogicalResult matchAndRewrite(triton::MatrixLoadOp op,OpAdaptor a,ConversionPatternRewriter &rw) const override {
    if(!op->hasAttr(StructuredRegisterRequest))return failure();
    auto proof=op->getAttrOfType<IntegerAttr>(StructuredRegisterWidth);
    if(!proof)return op.emitError("structured-register: same-pass preflight witness missing");
    unsigned width=proof.getInt();
    if(width!=1 && width!=2 && width!=4 && width!=8)return op.emitError("structured-register: invalid vector witness");
    // K/strides/layout have been validated before any type conversion.
    int64_t K=op.getTensorShape()[1];
    auto kAttr=op.getShape()[1].getDefiningOp<arith::ConstantOp>();
    if(kAttr)K=cast<IntegerAttr>(kAttr.getValue()).getInt();
    else if(auto c=a.getShape()[1].getDefiningOp<LLVM::ConstantOp>())K=cast<IntegerAttr>(c.getValue()).getInt();
    else return op.emitError("structured-register: proved K constant was lost");
    auto &rewriter=rw;auto loc=op.getLoc();auto b=TritonLLVMOpBuilder(loc,rw);auto *ctx=rw.getContext();
    auto elem=getTypeConverter()->convertType(op.getType().getElementType());
    Value M=a.getShape()[0],R=a.getIndices()[0],C=a.getIndices()[1];
    Value safeM=b.select(b.icmp_sgt(M,b.i32_val(0)),M,b.i32_val(0));
    Value nonnegative=b.select(b.icmp_sge(R,b.i32_val(0)),R,b.i32_val(0));
    Value rowBase=b.select(b.icmp_ult(nonnegative,safeM),nonnegative,safeM);
    Value validK=b.icmp_ule(C,b.i32_val(K-op.getType().getShape()[1]));
    auto unclampedAttr=op->getParentOfType<ModuleOp>()->getAttrOfType<BoolAttr>(
        "ttg.hcu.structured_register_unclamped_column");
    // The existing validK/enable guard still makes NR=0 for every invalid
    // column. The descriptor base then has no effective memory access.
    // No inbounds/nowrap promises are added to its integer/pointer math.
    Value colBase=unclampedAttr&&unclampedAttr.getValue()?C:b.select(validK,C,b.i32_val(0));
    Value remaining=mlir::triton::HCU::memory_analysis::uniformSaturatingSub(loc,rw,safeM,rowBase);
    Value enabled=b.and_(validK,b.icmp_ne(a.getIndices()[2],b.i32_val(0)));
    Value records=b.select(enabled,remaining,b.i32_val(0));
    Value baseOffset=b.sub(b.add(b.mul(b.zext(i64_ty,rowBase),b.i64_val(K)),b.zext(i64_ty,colBase)),b.i64_val(8));
    Value base=b.gep(ptr_ty(ctx,1),elem,a.getBase(),baseOffset);
    auto resourceFn=getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rw,
        "llvm.amdgcn.make.buffer.rsrc.p8.p1",ptr_ty(ctx,8),{ptr_ty(ctx,1),i16_ty,i32_ty,i32_ty});
    auto prepackAttr=op->getParentOfType<ModuleOp>()->getAttrOfType<BoolAttr>(
        "ttg.hcu.structured_register_prepack_address");
    bool prepack = prepackAttr && prepackAttr.getValue();
    Value resource = LLVM::createLLVMCallOp(
        rw, loc, resourceFn,
        {base, b.int_val(16, 2 * K | 0x4000), records, b.i32_val(131072)})
                         .getResult();
    Type payload;
    if(width==1)payload=i16_ty;else if(width==2)payload=i32_ty;else payload=vec_ty(i32_ty,width/2);
    std::string suffix=width==1?"i16":width==2?"i32":width==4?"v2i32":"v4i32";
    auto loadFn = getOrInsertHCUIntrinsic(op->getParentOfType<ModuleOp>(),loc,rw,
        "llvm.amdgcn.struct.ptr.buffer.load."+suffix,payload,{ptr_ty(ctx,8),i32_ty,i32_ty,i32_ty,i32_ty});
    auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),warp=StringAttr::get(ctx,"warp");
    auto ll=triton::gpu::toLinearLayout(op.getType());ll=ll.sublayout({reg,lane,warp},llvm::to_vector(ll.getOutDimNames()));
    auto [laneId,warpId]=getLaneAndWarpId(rw,loc);
    SmallVector<Value> values(ll.getInDimSize(reg));
    auto order=op->getAttrOfType<DenseI32ArrayAttr>(StructuredRegisterOrder);
    auto original=[&](unsigned i) {return order?unsigned(order[i]):i;};
    for(unsigned i=0;i<ll.getInDimSize(reg);i+=width) {
      unsigned r=original(i);
      auto xy=applyLinearLayout(loc,rw,ll,{{reg,b.i32_val(r)},{lane,laneId},{warp,warpId}});
      // For negative R, unsigned rowIndex is OOB until r+R becomes nonnegative.
      // For R>M, records==0. For enabled in-range R, this is exactly local row.
      // All meaningful sums fit signed32; disabled records never dereference.
      Value index=b.add(xy[0].second,b.sub(R,rowBase));
      Value offset=b.add(b.shl(xy[1].second,b.i32_val(1)),b.i32_val(16));
      if(prepack) {
        // Bind the two halves as one live VReg64 address value before the
        // structured intrinsic extracts them again. This is a pure exact
        // register identity, with no memory clobber or synchronization.
        // Loop-invariant pairs may then be materialized before the loop.
        Type pairType=vec_ty(i32_ty,2);
        Value pair=LLVM::PoisonOp::create(rw,loc,pairType);
        pair=b.insert_element(pairType,pair,index,b.i32_val(0));
        pair=b.insert_element(pairType,pair,offset,b.i32_val(1));
        pair=LLVM::InlineAsmOp::create(rw,loc,pairType,ValueRange{pair},
            "","=v,0",false,false,LLVM::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx,LLVM::AsmDialect::AD_ATT),
            ArrayAttr::get(ctx,{})).getRes();
        index=b.extract_element(i32_ty,pair,b.i32_val(0));
        offset=b.extract_element(i32_ty,pair,b.i32_val(1));
      }
      Value loaded = LLVM::createLLVMCallOp(
          rw, loc, loadFn, {resource, index, offset, b.i32_val(0), b.i32_val(0)})
                         .getResult();
      if(width==1)values[original(i)]=b.bitcast(loaded,elem);
      else {
        Value casted=b.bitcast(loaded,vec_ty(elem,width));
        for(unsigned j=0;j<width;++j)
          values[original(i+j)]=b.extract_element(elem,casted,b.i32_val(j));
      }
    }
    rw.replaceOp(op,packLLElements(loc,getTypeConverter(),values,rw,op.getType()));return success();
  }
};
} // namespace
