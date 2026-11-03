// Included inside rawws. An unknown output condition is never interpreted as a
// queue/publication event. Only a terminal, global-store-only suffix may bypass
// condition evaluation; its complete two-sided graph is checked independently.
static StringRef callee(Operation* op);
static bool knownResourceConstructor(Operation* op);
static bool knownFPClass(Operation* op);
static bool knownGlobalI16Store(Operation* op);

static bool earlySuffixType(Type type,unsigned depth=0) {
  if(depth>16)return false;
  if(isa<IntegerType,IndexType,FloatType>(type))return true;
  if(auto tensor=dyn_cast<RankedTensorType>(type))return earlySuffixType(tensor.getElementType(),depth+1);
  if(auto ptr=dyn_cast<triton::PointerType>(type))return ptr.getAddressSpace()==1;
  return false; // In particular, no memdesc/token/shared pointer.
}
static bool earlySuffixGlobalStore(Operation* op) {
  if(op->getNumRegions()||op->getNumSuccessors()||op->getNumResults())return false;
  Value pointer;
  if(auto store=dyn_cast<triton::StoreOp>(op))pointer=store.getPtr();
  else if(auto store=dyn_cast<triton::hcugpu::BufferStoreOp>(op))pointer=store.getPtr();
  else return false;
  Type t=pointer.getType();if(auto tensor=dyn_cast<RankedTensorType>(t))t=tensor.getElementType();
  auto ptr=dyn_cast<triton::PointerType>(t);
  return ptr&&ptr.getAddressSpace()==1&&llvm::all_of(op->getOperandTypes(),[](Type t){return earlySuffixType(t);});
}
static bool earlySuffixPure(Operation* op) {
  if(op->getNumRegions()||op->getNumSuccessors()||op->hasTrait<OpTrait::IsTerminator>()||!isPure(op))return false;
  const StringRef allowed[]={
    "arith.constant","arith.addi","arith.subi","arith.muli","arith.divsi","arith.divui","arith.remsi","arith.remui",
    "arith.andi","arith.ori","arith.xori","arith.shli","arith.shrsi","arith.shrui","arith.cmpi","arith.cmpf","arith.select",
    "arith.extsi","arith.extui","arith.trunci","arith.index_cast","arith.index_castui","arith.bitcast",
    "arith.addf","arith.subf","arith.mulf","arith.divf","arith.extf","arith.truncf","arith.sitofp","arith.uitofp","arith.fptosi","arith.fptoui",
    "tt.get_program_id","tt.make_range","tt.splat","tt.broadcast","tt.expand_dims","tt.addptr","tt.fp_to_fp"};
  // convert_layout, dot/MMA, shared operations and all inline asm stay excluded,
  // even when an interface advertises purity. Late proof must see no LDS code.
  return llvm::is_contained(allowed,op->getName().getStringRef())&&
      llvm::all_of(op->getOperandTypes(),[](Type t){return earlySuffixType(t);})&&
      llvm::all_of(op->getResultTypes(),[](Type t){return earlySuffixType(t);});
}
// Store suffixes do not grant permission for lane/wave-varying control.
// Unlike the finite actor evaluator, this check accepts unknown CTA program IDs
// and integer entry arguments but rejects loop PHIs, readfirstlane/tid and loads.
static bool earlySuffixUniform(Value v,llvm::SmallPtrSetImpl<void*>& active,unsigned depth=0) {
  if(depth>64||!v.getType().isIntOrIndex())return false;
  Value c=capture(v);if(c!=v)return earlySuffixUniform(c,active,depth+1);
  if(auto arg=dyn_cast<BlockArgument>(v)) {
    auto function=dyn_cast<triton::FuncOp>(arg.getOwner()->getParentOp());
    return function&&!function.getBody().empty()&&arg.getOwner()==&function.getBody().front();
  }
  Operation* op=v.getDefiningOp();if(!op||!active.insert(v.getAsOpaquePointer()).second)return false;
  auto erase=llvm::make_scope_exit([&]{active.erase(v.getAsOpaquePointer());});
  if(auto constant=dyn_cast<arith::ConstantOp>(op))return bool(dyn_cast<IntegerAttr>(constant.getValue()));
  if(isa<triton::GetProgramIdOp>(op))return op->getNumOperands()==0&&op->getNumResults()==1&&v.getType().isInteger(32);
  if(!op->getNumOperands()||op->getNumResults()!=1||!earlySuffixPure(op))return false;
  return llvm::all_of(op->getOperands(),[&](Value x){return earlySuffixUniform(x,active,depth+1);});
}
static bool earlyStoreOnlySuffix(scf::IfOp branch) {
  if(branch.getNumResults()||!branch.getCondition().getType().isInteger(1))return false;
  llvm::SmallPtrSet<void*,32> condition;
  if(!earlySuffixUniform(branch.getCondition(),condition))return false;
  auto ws=branch->getParentOfType<WarpSpecializeOp>();
  // Only the actual default-region terminal if, not a loop-body or nested if.
  if(!ws||branch->getParentRegion()!=&ws.getDefaultRegion()||!llvm::hasSingleElement(ws.getDefaultRegion()))return false;
  Operation* next=branch->getNextNode();
  if(!next||!isa<WarpYieldOp,WarpReturnOp>(next)||next->getNumOperands()||next->getNextNode())return false;
  unsigned budget=0;
  for(Region* region:{&branch.getThenRegion(),&branch.getElseRegion()}) {
    if(region->empty())continue;
    if(!llvm::hasSingleElement(*region)||region->front().getNumArguments())return false;
    Block& block=region->front();auto yield=dyn_cast<scf::YieldOp>(block.getTerminator());
    if(!yield||yield->getNumOperands())return false;
    for(Operation& op:block) {
      if(++budget>65536)return false;
      if(&op==yield.getOperation())continue;
      if(!earlySuffixPure(&op)&&!earlySuffixGlobalStore(&op))return false;
    }
  }
  return true;
}

static bool lateSuffixType(Type type,unsigned depth=0) {
  if(depth>16)return false;
  if(isa<IntegerType,IndexType,FloatType>(type))return true;
  if(auto ptr=dyn_cast<LLVM::LLVMPointerType>(type))return ptr.getAddressSpace()==0||ptr.getAddressSpace()==1||ptr.getAddressSpace()==8;
  if(auto vector=dyn_cast<VectorType>(type))return lateSuffixType(vector.getElementType(),depth+1);
  if(auto array=dyn_cast<LLVM::LLVMArrayType>(type))return lateSuffixType(array.getElementType(),depth+1);
  if(auto structure=dyn_cast<LLVM::LLVMStructType>(type))
    return !structure.isOpaque()&&llvm::all_of(structure.getBody(),[&](Type t){return lateSuffixType(t,depth+1);});
  return false;
}
static bool suffixZeroAux(Value v) {
  auto constant=v.getDefiningOp<LLVM::ConstantOp>();
  auto value=constant?dyn_cast<IntegerAttr>(constant.getValue()):IntegerAttr{};
  return value&&value.getType().isInteger(32)&&value.getInt()==0;
}
static bool lateSuffixGlobalStore(Operation* op) {
  if(op->getNumRegions()||op->getNumSuccessors()||op->getNumResults())return false;
  if(isa<ROCDL::RawPtrBufferStoreOp>(op))return knownGlobalI16Store(op)&&suffixZeroAux(op->getOperand(4));
  if(auto store=dyn_cast<LLVM::StoreOp>(op)) {
    auto ptr=dyn_cast<LLVM::LLVMPointerType>(store.getAddr().getType());
    // Conservative ordinary-store subset; atomic/volatile forms are not a
    // publication-free arithmetic suffix, even when the destination is AS1.
    return ptr&&ptr.getAddressSpace()==1&&!op->hasAttr("volatile_")&&!op->hasAttr("volatile")&&
        !op->hasAttr("ordering")&&!op->hasAttr("syncscope")&&
        llvm::all_of(op->getOperandTypes(),[](Type t){return lateSuffixType(t);});
  }
  if(!isa<LLVM::CallOp>(op)||callee(op)!="llvm.amdgcn.raw.ptr.buffer.store.i16"||op->getNumOperands()!=5)return false;
  auto resource=dyn_cast<LLVM::LLVMPointerType>(op->getOperand(1).getType());
  return op->getOperand(0).getType().isInteger(16)&&resource&&resource.getAddressSpace()==8&&
      op->getOperand(2).getType().isInteger(32)&&op->getOperand(3).getType().isInteger(32)&&
      op->getOperand(4).getType().isInteger(32)&&suffixZeroAux(op->getOperand(4));
}
static bool lateSuffixPure(Operation* op) {
  if(op->getNumRegions()||op->getNumSuccessors()||op->hasTrait<OpTrait::IsTerminator>())return false;
  if(!llvm::all_of(op->getOperandTypes(),[](Type t){return lateSuffixType(t);})||
     !llvm::all_of(op->getResultTypes(),[](Type t){return lateSuffixType(t);}))return false;
  if(isa<LLVM::CallIntrinsicOp>(op))return knownFPClass(op);
  if(isa<ROCDL::MakeBufferRsrcOp>(op)||isa<LLVM::CallOp>(op))return knownResourceConstructor(op);
  if(!isPure(op))return false;
  const StringRef allowed[]={
    "llvm.mlir.constant","llvm.mlir.undef","llvm.mlir.poison",
    "llvm.add","llvm.sub","llvm.mul","llvm.udiv","llvm.sdiv","llvm.urem","llvm.srem",
    "llvm.and","llvm.or","llvm.xor","llvm.shl","llvm.lshr","llvm.ashr","llvm.icmp","llvm.fcmp","llvm.select",
    "llvm.zext","llvm.sext","llvm.trunc","llvm.bitcast","llvm.getelementptr","llvm.ptrtoint","llvm.inttoptr","llvm.addrspacecast",
    "llvm.fadd","llvm.fsub","llvm.fmul","llvm.fdiv","llvm.fpext","llvm.fptrunc","llvm.sitofp","llvm.uitofp","llvm.fptosi","llvm.fptoui",
    "llvm.extractelement","llvm.insertelement","llvm.shufflevector","llvm.extractvalue","llvm.insertvalue",
    "arith.constant","arith.addi","arith.subi","arith.muli","arith.cmpi","arith.select","arith.extui","arith.extsi","arith.trunci","arith.index_cast"};
  return llvm::is_contained(allowed,op->getName().getStringRef());
}
static bool lateSuffixUniform(Value v,llvm::SmallPtrSetImpl<void*>& active,unsigned depth=0) {
  if(depth>64||!v.getType().isIntOrIndex())return false;
  Value c=capture(v);if(c!=v)return lateSuffixUniform(c,active,depth+1);
  if(auto arg=dyn_cast<BlockArgument>(v)) {
    auto function=dyn_cast<LLVM::LLVMFuncOp>(arg.getOwner()->getParentOp());
    return function&&!function.getBody().empty()&&arg.getOwner()==&function.getBody().front();
  }
  Operation* op=v.getDefiningOp();if(!op||!active.insert(v.getAsOpaquePointer()).second)return false;
  auto erase=llvm::make_scope_exit([&]{active.erase(v.getAsOpaquePointer());});
  if(auto constant=dyn_cast<LLVM::ConstantOp>(op))return bool(dyn_cast<IntegerAttr>(constant.getValue()));
  if(auto constant=dyn_cast<arith::ConstantOp>(op))return bool(dyn_cast<IntegerAttr>(constant.getValue()));
  StringRef n=op->getName().getStringRef();
  if(n=="rocdl.workgroup.id.x"||n=="rocdl.workgroup.id.y")
    return op->getNumOperands()==0&&op->getNumResults()==1&&v.getType().isInteger(32);
  if(!op->getNumOperands()||op->getNumResults()!=1||!lateSuffixPure(op))return false;
  return llvm::all_of(op->getOperands(),[&](Value x){return lateSuffixUniform(x,active,depth+1);});
}
// Prove ALL paths from a particular unknown terminal branch. Block arguments
// remain ordinary SSA: no values, conditions or queue state are invented.
static bool lateStoreOnlySuffix(Operation* root,Region& region) {
  if(root->getBlock()->getParent()!=&region||root!=root->getBlock()->getTerminator())return false;
  if(!isa<LLVM::CondBrOp,cf::CondBranchOp>(root))return false;
  Value rootCondition=isa<LLVM::CondBrOp>(root)?cast<LLVM::CondBrOp>(root).getCondition():cast<cf::CondBranchOp>(root).getCondition();
  llvm::SmallPtrSet<void*,32> condition;
  if(!lateSuffixUniform(rootCondition,condition))return false;
  llvm::DenseMap<Block*,unsigned> colors;
  llvm::SmallPtrSet<Block*,32> visited;
  unsigned operations=0;bool hasExit=false;
  std::function<bool(Block*,unsigned)> visit;
  auto successors=[&](Operation* terminator,SmallVectorImpl<Block*>& dests)->bool {
    if(auto br=dyn_cast<LLVM::BrOp>(terminator)) {
      if(br.getDestOperands().size()!=br.getDest()->getNumArguments())return false;
      dests.push_back(br.getDest());return true;
    }
    if(auto br=dyn_cast<LLVM::CondBrOp>(terminator)) {
      llvm::SmallPtrSet<void*,32> condition;
      if(!lateSuffixUniform(br.getCondition(),condition))return false;
      if(!br.getCondition().getType().isInteger(1)||br.getTrueDestOperands().size()!=br.getTrueDest()->getNumArguments()||br.getFalseDestOperands().size()!=br.getFalseDest()->getNumArguments())return false;
      dests.append({br.getTrueDest(),br.getFalseDest()});return true;
    }
    if(auto br=dyn_cast<cf::BranchOp>(terminator)) {
      if(br.getDestOperands().size()!=br.getDest()->getNumArguments())return false;
      dests.push_back(br.getDest());return true;
    }
    if(auto br=dyn_cast<cf::CondBranchOp>(terminator)) {
      llvm::SmallPtrSet<void*,32> condition;
      if(!lateSuffixUniform(br.getCondition(),condition))return false;
      if(!br.getCondition().getType().isInteger(1)||br.getTrueDestOperands().size()!=br.getTrueDest()->getNumArguments()||br.getFalseDestOperands().size()!=br.getFalseDest()->getNumArguments())return false;
      dests.append({br.getTrueDest(),br.getFalseDest()});return true;
    }
    return false;
  };
  visit=[&](Block* block,unsigned depth)->bool {
    if(!block||block->getParent()!=&region||block==root->getBlock()||depth>128)return false;
    unsigned mark=colors.lookup(block);if(mark)return mark==2; // Gray is a cycle.
    if(visited.size()>=4096)return false;
    colors[block]=1;visited.insert(block);
    if(!llvm::all_of(block->getArgumentTypes(),[](Type t){return lateSuffixType(t);}))return false;
    for(Operation& op:*block) {
      if(++operations>65536)return false;
      if(&op==block->getTerminator()) {
        if(isa<WarpYieldOp,WarpReturnOp>(&op)) {
          if(op.getNumOperands()||op.getNumResults()||op.getNumRegions()||op.getNumSuccessors())return false;
          hasExit=true;colors[block]=2;return true;
        }
        if(!llvm::all_of(op.getOperandTypes(),[](Type t){return lateSuffixType(t);}))return false;
        SmallVector<Block*> dests;if(!successors(&op,dests))return false;
        for(Block* dest:dests)if(!visit(dest,depth+1))return false;
        colors[block]=2;return true;
      }
      if(!lateSuffixPure(&op)&&!lateSuffixGlobalStore(&op))return false;
    }
    return false;
  };
  SmallVector<Block*> starts;if(!successors(root,starts))return false;
  for(Block* start:starts)if(!visit(start,0))return false;
  // Shared merges are fine, but no unexamined predecessor may enter the suffix.
  for(Block* block:visited)for(Block* pred:block->getPredecessors())
    if(pred!=root->getBlock()&&!visited.contains(pred))return false;
  return hasExit;
}
