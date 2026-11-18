#ifndef HCU_TLE_STATIC_PIPE_LLVM_H
#define HCU_TLE_STATIC_PIPE_LLVM_H

#include "TritonHCUGPUToLLVM/StaticPipe.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir::triton::HCU::static_pipe {
struct Frame {
  Value state, status, leader, wave;
  Block *exit=nullptr;
  bool dropReady=false;
};
inline Value unwrapLLVM(Value v) {
  while (auto c=v.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (c->getNumOperands()!=1) break;
    v=c->getOperand(0);
  }
  return v;
}
inline Value cell(TritonLLVMIRRewriter &b,Value base,Value index) {
  return b.gep(LLVM::LLVMPointerType::get(b.getContext(),3),b.getI32Type(),base,index);
}
inline Value cell(TritonLLVMIRRewriter &b,Value base,int index) {
  return cell(b,base,b.i32_val(index));
}
inline Value atomicLoad(TritonLLVMIRRewriter &b,Value ptr,LLVM::AtomicOrdering order) {
  auto op=b.load(b.getI32Type(),ptr,4);op.setOrdering(order);
  op.setSyncscope(StringRef("workgroup-one-as"));return op.getResult();
}
inline void atomicStore(TritonLLVMIRRewriter &b,Value ptr,Value value,
                        LLVM::AtomicOrdering order,bool global=false) {
  auto op=b.store(value,ptr,4);op.setOrdering(order);
  op.setSyncscope(StringRef(global ? "agent-one-as" : "workgroup-one-as"));
}
inline void fence(TritonLLVMIRRewriter &b,LLVM::AtomicOrdering order) {
  b.create<LLVM::FenceOp>(TypeRange{},order,b.getStringAttr("workgroup-one-as"));
}
inline void drain(TritonLLVMIRRewriter &b) { b.create<ROCDL::SWaitcntOp>(0); }

inline LogicalResult initialize(LLVM::LLVMFuncOp func,te::PipeCreateOp create,
                                 Block *exit,Frame &f) {
  auto mod=func->getParentOfType<ModuleOp>();
  auto attr=mod->getAttrOfType<IntegerAttr>("tle.hcu.static_pipe_status_arg");
  if (!attr || attr.getInt()<0 || attr.getInt()>=func.getNumArguments() || (create.getFields().size()!=2 && create.getFields().size()!=3 && create.getFields().size()!=5))
    return create.emitError("missing validated static pipe error channel");
  Value llvmState=unwrapLLVM(create.getFields().back());
  if (!isa<LLVM::LLVMStructType>(llvmState.getType()))
    return create.emitError("static pipe epoch memdesc was not converted");
  auto loc=create.getLoc();TritonLLVMIRRewriter b(loc,create);
  auto obj=LLVM::getSharedMemoryObjectFromStruct(loc,llvmState,b.getI32Type(),b);
  f.state=obj.getBase();f.exit=exit;
  auto test=mod->getAttrOfType<IntegerAttr>("tle.hcu.static_pipe_test_drop_ready");
  f.dropReady=test && test.getInt()!=0;
  Value tid=b.create<ROCDL::ThreadIdXOp>(b.getI32Type());
  Value lane=b.and_(tid,b.i32_val(63));f.leader=b.icmp_eq(lane,b.i32_val(0));
  f.wave=b.create<ROCDL::ReadfirstlaneOp>(b.getI32Type(),b.and_(b.udiv(tid,b.i32_val(64)),b.i32_val(1)));
  // HCU ROCDL grid dimensions translate through size_t OCKL helpers (i64).
  // Do not ask the generic index32 conversion to declare those values i32.
  Value x=b.zext(b.getI64Type(),b.create<ROCDL::BlockIdXOp>(b.getI32Type()));
  Value y=b.zext(b.getI64Type(),b.create<ROCDL::BlockIdYOp>(b.getI32Type()));
  Value z=b.zext(b.getI64Type(),b.create<ROCDL::BlockIdZOp>(b.getI32Type()));
  Value nx=b.create<ROCDL::GridDimXOp>(b.getI64Type());
  Value ny=b.create<ROCDL::GridDimYOp>(b.getI64Type());
  Value id=b.add(x,b.mul(nx,b.add(y,b.mul(ny,z))));
  f.status=b.gep(LLVM::LLVMPointerType::get(b.getContext(),1),b.getI32Type(),func.getArgument(attr.getInt()),id);
  Value first=b.icmp_eq(tid,b.i32_val(0));
  Block *before=create->getBlock(),*after=b.splitBlock(before,create->getIterator());
  Block *init=b.createBlock(after);b.setInsertionPointToEnd(before);
  b.create<LLVM::CondBrOp>(first,init,after);
  b.setInsertionPointToStart(init);
  for (int i=0;i<16;++i) atomicStore(b,cell(b,f.state,i),b.i32_val(0),LLVM::AtomicOrdering::monotonic);
  atomicStore(b,f.status,b.i32_val(0),LLVM::AtomicOrdering::monotonic,true);
  b.create<LLVM::BrOp>(ValueRange{},after);
  b.setInsertionPointToStart(after);drain(b);fence(b,LLVM::AtomicOrdering::release);
  b.create<ROCDL::BarrierOp>();fence(b,LLVM::AtomicOrdering::acquire);
  create.erase();return success();
}

// A complete wave takes the same success/failure branch. Only lane0 polls;
// after reconvergence every lane performs acquire loads before payload access.
inline void waitPair(Operation *op,Frame &f,Value slot,Value expected,bool ready) {
  auto loc=op->getLoc();TritonLLVMIRRewriter b(loc,op);
  Value a=cell(b,f.state,b.add(b.mul(slot,b.i32_val(4)),b.i32_val(ready?0:2))),c=cell(b,f.state,b.add(b.mul(slot,b.i32_val(4)),b.i32_val(ready?1:3)));
  Value abort=cell(b,f.state,8),target=expected;
  Block *before=op->getBlock(),*after=b.splitBlock(before,op->getIterator());
  auto make=[&]() {return b.createBlock(after);};
  Block *poll=make(),*failedCheck=make(),*again=make(),*success=make(),*failed=make(),*join=make(),*acquire=make();
  poll->addArgument(b.getI32Type(),loc);join->addArgument(b.getI32Type(),loc);
  b.setInsertionPointToEnd(before);
  b.create<LLVM::CondBrOp>(f.leader,poll,ValueRange{b.i32_val(0)},join,ValueRange{b.i32_val(0)});
  b.setInsertionPointToStart(poll);
  Value va=atomicLoad(b,a,LLVM::AtomicOrdering::acquire),vc=atomicLoad(b,c,LLVM::AtomicOrdering::acquire);
  Value ba=atomicLoad(b,abort,LLVM::AtomicOrdering::monotonic);
  Value ok=b.and_(b.and_(b.icmp_eq(va,target),b.icmp_eq(vc,target)),b.icmp_eq(ba,b.i32_val(0)));
  b.create<LLVM::CondBrOp>(ok,success,failedCheck);
  b.setInsertionPointToStart(failedCheck);
  Value stop=b.or_(b.icmp_ne(ba,b.i32_val(0)),b.icmp_uge(poll->getArgument(0),b.i32_val(19999)));
  b.create<LLVM::CondBrOp>(stop,failed,again);
  b.setInsertionPointToStart(again);
  LLVM::createLLVMIntrinsicCallOp(b,loc,"llvm.amdgcn.s.sleep",{},ValueRange{b.i32_val(1)});
  b.create<LLVM::BrOp>(ValueRange{b.add(poll->getArgument(0),b.i32_val(1))},poll);
  b.setInsertionPointToStart(success);b.create<LLVM::BrOp>(ValueRange{b.i32_val(1)},join);
  b.setInsertionPointToStart(failed);
  atomicStore(b,abort,b.i32_val(1),LLVM::AtomicOrdering::release);
  atomicStore(b,f.status,b.i32_val(1),LLVM::AtomicOrdering::release,true);
  b.create<LLVM::BrOp>(ValueRange{b.i32_val(0)},join);
  b.setInsertionPointToStart(join);
  Value uniform=b.create<ROCDL::ReadlaneOp>(b.getI32Type(),join->getArgument(0),b.i32_val(0));
  b.create<LLVM::CondBrOp>(b.icmp_ne(uniform,b.i32_val(0)),acquire,f.exit);
  b.setInsertionPointToStart(acquire);
  Value finalA=atomicLoad(b,a,LLVM::AtomicOrdering::acquire),finalC=atomicLoad(b,c,LLVM::AtomicOrdering::acquire);
  Value all=b.create<ROCDL::BallotOp>(b.getI64Type(),b.and_(b.icmp_eq(finalA,target),b.icmp_eq(finalC,target)));
  fence(b,LLVM::AtomicOrdering::acquire);
  b.create<LLVM::CondBrOp>(b.icmp_eq(all,b.i64_val(-1)),after,failed);
  if (op->getNumResults()) {
    b.setInsertionPointToStart(after);op->getResult(0).replaceAllUsesWith(b.false_val());
  }
  op->erase();
}
inline void publish(Operation *op,Frame &f,Value slot,Value epoch,bool ready) {
  auto loc=op->getLoc();TritonLLVMIRRewriter b(loc,op);
  Value abort=atomicLoad(b,cell(b,f.state,8),LLVM::AtomicOrdering::monotonic);
  abort=b.create<ROCDL::ReadfirstlaneOp>(b.getI32Type(),abort);
  Value ptr=cell(b,f.state,b.add(b.add(b.mul(slot,b.i32_val(4)),b.i32_val(ready?0:2)),f.wave));
  Block *before=op->getBlock(),*after=b.splitBlock(before,op->getIterator());
  Block *live=b.createBlock(after),*leader=b.createBlock(after);
  b.setInsertionPointToEnd(before);b.create<LLVM::CondBrOp>(b.icmp_ne(abort,b.i32_val(0)),f.exit,live);
  b.setInsertionPointToStart(live);drain(b);fence(b,LLVM::AtomicOrdering::release);
  Value elected=f.leader;
  if (ready && f.dropReady) elected=b.and_(elected,b.icmp_eq(f.wave,b.i32_val(0)));
  b.create<LLVM::CondBrOp>(elected,leader,after);
  b.setInsertionPointToStart(leader);
  atomicStore(b,ptr,epoch,LLVM::AtomicOrdering::release);
  b.create<LLVM::BrOp>(ValueRange{},after);op->erase();
}
inline LogicalResult lower(LLVM::LLVMFuncOp func,Block *exit) {
  te::PipeCreateOp create;SmallVector<Operation*> ops;
  func.walk([&](Operation *op) {if (auto c=dyn_cast<te::PipeCreateOp>(op)) create=c;else if (isPipe(op)) ops.push_back(op);});
  if (!create) return success();
  Frame f;if (failed(initialize(func,create,exit,f))) return failure();
  llvm::DenseMap<int,std::pair<Value,Value>> phases;
  for (Operation *op:ops) if (auto pair=op->getAttrOfType<IntegerAttr>(pairAttr)) {
    if (auto ac=dyn_cast<te::PipeWriterAcquireOp>(op)) phases[pair.getInt()]={unwrapLLVM(ac.getStage()),unwrapLLVM(ac.getPhase())};
    if (auto ac=dyn_cast<te::PipeReaderWaitOp>(op)) phases[pair.getInt()]={unwrapLLVM(ac.getStage()),unwrapLLVM(ac.getPhase())};
  }
  for (Operation *op:ops) {
    if (auto pair=op->getAttrOfType<IntegerAttr>(pairAttr)) {
      auto it=phases.find(pair.getInt());
      if (it==phases.end()) return op->emitError("missing counted pipe stage/phase source");
      TritonLLVMIRRewriter b(op->getLoc(),op);
      auto [slot,phaseBit]=it->second;
      if (!slot.getType().isInteger(32) || !phaseBit.getType().isInteger(1))
        return op->emitError("counted pipe phase conversion is invalid");
      Value phase=b.zext(b.getI32Type(),phaseBit);
      Value complete=b.xor_(phase,b.i32_val(1));
      if (isa<te::PipeWriterAcquireOp>(op)) waitPair(op,f,slot,phase,false);
      else if (isa<te::PipeReaderWaitOp>(op)) waitPair(op,f,slot,complete,true);
      else if (isa<te::PipeWriterCommitOp>(op)) publish(op,f,slot,complete,true);
      else if (isa<te::PipeReaderReleaseOp>(op)) publish(op,f,slot,complete,false);
      else return op->emitError("unsupported counted pipe lifecycle");
      continue;
    }
    auto index=op->getAttrOfType<IntegerAttr>(iterationAttr);
    if (!index || index.getInt()<0 || index.getInt()>=16) return op->emitError("missing static pipe iteration proof");
    int k=index.getInt();TritonLLVMIRRewriter b(op->getLoc(),op);
    Value slot=b.i32_val(k%2),epoch=b.i32_val(k/2+1);
    if (isa<te::PipeWriterAcquireOp>(op)) waitPair(op,f,slot,b.i32_val(k/2),false);
    else if (isa<te::PipeReaderWaitOp>(op)) waitPair(op,f,slot,epoch,true);
    else if (isa<te::PipeWriterCommitOp>(op)) publish(op,f,slot,epoch,true);
    else if (isa<te::PipeReaderReleaseOp>(op)) publish(op,f,slot,epoch,false);
    else return op->emitError("unsupported static pipe lifecycle");
  }
  return success();
}
} // namespace mlir::triton::HCU::static_pipe
#endif
