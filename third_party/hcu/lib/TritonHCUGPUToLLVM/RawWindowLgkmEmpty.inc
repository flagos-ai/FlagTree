// Included in namespace rawws after the existing exact late-op helpers.
// A must property of this wave's LGKM counter, never a cross-wave race proof.
static std::optional<int64_t> constantWaitMask(Operation* op) {
  if(auto wait=dyn_cast<ROCDL::SWaitcntOp>(op)) {
    auto field=wait.getBitfieldAttr();
    if(op->getNumOperands()||op->getNumResults()||!field||!field.getType().isInteger(32))return {};
    int64_t n=field.getInt();return n>=0&&n<=65535?std::optional<int64_t>(n):std::nullopt;
  }
  if(!isa<LLVM::CallOp>(op)||callee(op)!="llvm.amdgcn.s.waitcnt"||op->getNumOperands()!=1||op->getNumResults()||!op->getOperand(0).getType().isInteger(32))return {};
  Operation* c=op->getOperand(0).getDefiningOp();if(!c)return {};
  StringRef name=c->getName().getStringRef();
  if(name!="llvm.mlir.constant"&&name!="arith.constant")return {};
  auto value=c->getAttrOfType<IntegerAttr>("value");
  if(!value||!value.getType().isInteger(32))return {};
  int64_t n=value.getInt();return n>=0&&n<=65535?std::optional<int64_t>(n):std::nullopt;
}
static bool knownVMVectorGPRLoad(Operation* op) {
  if(knownRawRegisterGPRLoad(op))return true;
  if(!isa<LLVM::CallOp>(op)||callee(op)!="llvm.amdgcn.struct.ptr.buffer.load.v4i32"||op->getNumOperands()!=5||op->getNumResults()!=1)return false;
  auto ptr=dyn_cast<LLVM::LLVMPointerType>(op->getOperand(0).getType());
  auto vec=dyn_cast<VectorType>(op->getResult(0).getType());
  if(!ptr||ptr.getAddressSpace()!=8||!vec||vec.getRank()!=1||vec.getNumElements()!=4||!vec.getElementType().isInteger(32))return false;
  for(unsigned i=1;i<5;++i)if(!op->getOperand(i).getType().isInteger(32))return false;
  return true;
}
static bool knownVMRawCopy(Operation* op) {
  if(!isa<LLVM::CallOp>(op)||callee(op)!="llvm.hcu.raw.buffer.load.lds"||op->getNumOperands()!=7||op->getNumResults())return false;
  auto resource=dyn_cast<VectorType>(op->getOperand(0).getType());
  auto lds=dyn_cast<LLVM::LLVMPointerType>(op->getOperand(1).getType());
  if(!resource||resource.getRank()!=1||resource.getNumElements()!=4||!resource.getElementType().isInteger(32)||!lds||lds.getAddressSpace()!=3)return false;
  for(unsigned i=2;i<7;++i)if(!op->getOperand(i).getType().isInteger(32))return false;
  return true;
}
// The SDK ROCDL MMAC declaration lacks MemoryEffectOpInterface, while its
// LLVM HCUMMopIntrinsic explicitly has IntrNoMem. This exact registered op and
// complete signature establish only counter neutrality, never control/data
// uniformity or cross-wave synchronization. No shape/layout selector is used.
static bool knownRegisterOnlyBf16Mmac(Operation* op) {
  if(!isa<ROCDL::mmac_f32_16x16x16bf16>(op)||op->getNumRegions()||
     op->getNumSuccessors()||op->getNumOperands()!=5||op->getNumResults()!=1)
    return false;
  auto a=dyn_cast<VectorType>(op->getOperand(0).getType());
  auto b=dyn_cast<VectorType>(op->getOperand(1).getType());
  auto acc=dyn_cast<VectorType>(op->getOperand(2).getType());
  auto dst=dyn_cast<VectorType>(op->getResult(0).getType());
  return a&&b&&acc&&dst&&a.getRank()==1&&b.getRank()==1&&
    acc.getRank()==1&&dst.getRank()==1&&a.getNumElements()==4&&
    b.getNumElements()==4&&acc.getNumElements()==4&&dst.getNumElements()==4&&
    a.getElementType().isInteger(16)&&b.getElementType().isInteger(16)&&
    acc.getElementType().isF32()&&dst.getElementType().isF32()&&
    op->getOperand(3).getType().isInteger(1)&&
    op->getOperand(4).getType().isInteger(1);
}
static bool preservesEmptyLgkm(Operation* op) {
  StringRef n=op->getName().getStringRef(),fn=callee(op);
  if(isa<LLVM::LoadOp>(op))return false; // AS3, scalar, volatile or ordinary: conservative kill.
  if(auto store=dyn_cast<LLVM::StoreOp>(op)) {
    auto ptr=dyn_cast<LLVM::LLVMPointerType>(store.getAddr().getType());
    return ptr&&ptr.getAddressSpace()==1;
  }
  if(isa<ROCDL::SBarrierOp,ROCDL::BarrierOp,LLVM::BrOp,LLVM::CondBrOp,cf::BranchOp,cf::CondBranchOp,WarpYieldOp,WarpReturnOp>(op))return true;
  if(isa<LLVM::CallIntrinsicOp>(op))return knownFPClass(op);
  if(isa<ROCDL::MakeBufferRsrcOp>(op))return knownResourceConstructor(op);
  if(isa<ROCDL::RawPtrBufferStoreOp>(op))return knownGlobalI16Store(op);
  if(isa<LLVM::CallOp>(op)) {
    if(fn=="__triton_hcu_local_barrier_marker")return knownLocalBarrierMarker(op);
    if(fn=="llvm.amdgcn.make.buffer.rsrc.p8.p1")return knownResourceConstructor(op);
    if(fn=="llvm.amdgcn.workitem.id.x")return op->getNumOperands()==0&&op->getNumResults()==1&&op->getResult(0).getType().isInteger(32);
    if(fn=="llvm.amdgcn.readfirstlane")return op->getNumOperands()==1&&op->getNumResults()==1&&op->getOperand(0).getType().isInteger(32)&&op->getResult(0).getType().isInteger(32);
    return knownVMVectorGPRLoad(op)||knownVMRawCopy(op);
  }
  if(n=="llvm.inline_asm") {
    if(knownScalarClamp(op))return true;
    auto text=op->getAttrOfType<StringAttr>("asm_string");
    return text&&(text.getValue().empty()||text.getValue()=="s_setprio 0"||text.getValue()=="s_setprio 1");
  }
  if(n=="rocdl.readfirstlane"||n=="rocdl.workitem.id.x"||n=="gpu.thread_id"||knownWorkgroupQuery(op))return true;
  if(knownRegisterOnlyBf16Mmac(op))return true;
  if((n.starts_with("rocdl.mmac.")||n.starts_with("rocdl.mfma."))&&isMemoryEffectFree(op))return true;
  // Only registered, pure target MMA/arithmetic are counter-neutral. Unknown
  // calls/memory/wait/region operations never gain authority from a name alone.
  if(op->getNumRegions()||n.contains("call")||n.contains("load")||n.contains("store")||n.contains("wait")||n.contains("async")||n.contains("barrier"))return false;
  return isPure(op);
}
static bool emptyLgkmAfter(Operation* op,bool empty) {
  if(auto mask=constantWaitMask(op)) {
    if((*mask&0x0f00)==0)return true;
    return empty;
  }
  return preservesEmptyLgkm(op)?empty:false;
}
static void cleanupMustEmptyLgkm(WarpSpecializeOp ws) {
  auto mod=ws->getParentOfType<ModuleOp>();Builder attrs(ws.getContext());
  auto report=[&](unsigned deleted,StringRef why) {
    mod->setAttr("ttg.hcu.raw_window_ws.lgkm_empty_deleted",attrs.getI32IntegerAttr(deleted));
    mod->setAttr("ttg.hcu.raw_window_ws.lgkm_empty_cleanup",attrs.getStringAttr(why));
  };
  report(0,"not-applied");Region& region=ws.getDefaultRegion();
  if(region.empty()) {report(0,"empty-region");return;}
  llvm::SmallPtrSet<Block*,32> reachable;SmallVector<Block*> todo{&region.front()};
  llvm::DenseMap<Block*,SmallVector<Block*>> predecessors;
  while(!todo.empty()) {
    Block* block=todo.pop_back_val();if(!reachable.insert(block).second)continue;
    if(block->getParent()!=&region||block->empty()){report(0,"invalid-block");return;}
    Operation* term=&block->back();
    if(!term->hasTrait<OpTrait::IsTerminator>()){report(0,"missing-terminator");return;}
    if(!isa<LLVM::BrOp,LLVM::CondBrOp,cf::BranchOp,cf::CondBranchOp,WarpYieldOp,WarpReturnOp>(term)) {report(0,"unknown-control");return;}
    for(Block* successor:term->getSuccessors()) {
      if(successor->getParent()!=&region){report(0,"side-exit");return;}
      predecessors[successor].push_back(block);todo.push_back(successor);
    }
  }
  // Greatest fixed point for a MUST fact: entry unknown, all other reachable
  // initial outputs top=true; predecessor meet is AND. Reachable backedges do
  // not manufacture a fact without the non-loop incoming edge also proving it.
  llvm::DenseMap<Block*,bool> output;
  for(Block* b:reachable)output[b]=true;
  auto input=[&](Block* b) {
    if(b==&region.front()||predecessors[b].empty())return false;
    return llvm::all_of(predecessors[b],[&](Block* p){return output[p];});
  };
  bool changed=true;unsigned budget=0;
  while(changed) {
    changed=false;
    for(Block& b:region)if(reachable.contains(&b)) {
      bool state=input(&b);
      for(Operation& op:b) {
        if(++budget>1048576){report(0,"analysis-budget");return;}
        state=emptyLgkmAfter(&op,state);
      }
      if(output[&b]!=state){output[&b]=state;changed=true;}
    }
  }
  SmallVector<Operation*> erase;
  for(Block& b:region)if(reachable.contains(&b)) {
    bool state=input(&b);
    for(Operation& op:b) {
      auto mask=constantWaitMask(&op);
      // 0xC07F means VM63/EXP7/LGKM0. A full0 or VM0+LGKM0 wait
      // may establish empty, but must NEVER be deleted by this helper.
      if(mask&&*mask==0xc07f&&state)erase.push_back(&op);
      state=emptyLgkmAfter(&op,state);
    }
  }
  for(Operation* op:erase)op->erase();
  report(erase.size(),"applied-reachable-must-empty-c07f-only");
}
