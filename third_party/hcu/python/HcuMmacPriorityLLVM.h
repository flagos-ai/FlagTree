// Experimental gfx938 wave-local MMAC priority scopes. No arithmetic changes.
#pragma once
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Verifier.h"
#include <stdexcept>
namespace hcu_tle_priority {
using namespace llvm;
struct Report { unsigned groups=0,mmas=0; std::string reason; };
inline StringRef name(CallBase *c) {auto *f=c->getCalledFunction();return f?f->getName():StringRef();}
inline bool matrix(CallBase *c) {return name(c).starts_with("llvm.hcu.mmac.");}
inline Report run(Module &m,StringRef arch,unsigned waves,unsigned mode) {
  Report r;
  if(arch!="gfx938" || waves!=4 || mode<1 || mode>3) {r.reason="requires gfx938/four waves/mode1..3";return r;}
  SmallVector<SmallVector<CallInst *,8>,32> groups;
  for(auto &f:m)if(f.getCallingConv()==CallingConv::AMDGPU_KERNEL) {
    // Fail before mutation: never nest a second wave-priority policy.
    for(auto &bb:f)for(auto &i:bb)if(auto *c=dyn_cast<CallBase>(&i)) {
      if(name(c)=="llvm.amdgcn.s.setprio" || (c->isInlineAsm() &&
         cast<InlineAsm>(c->getCalledOperand())->getAsmString().find("s_setprio")!=std::string::npos)) {
        r.reason="existing wave-priority operation";return r;
      }
    }
    for(auto &bb:f) {
      SmallVector<CallInst *,8> current;
      auto flush=[&]() {if(!current.empty()){groups.push_back(current);current.clear();}};
      for(auto &i:bb) {
        if(auto *c=dyn_cast<CallInst>(&i)) {
          if(matrix(c)) {current.push_back(c);if(current.size()==8)flush();continue;}
          // Memory/sync/exp/other calls bound a cluster. Pure SSA packing and
          // integer address calculations are allowed between its MMAC nodes.
          flush();
        } else if(i.mayReadOrWriteMemory() || i.isTerminator())flush();
      }
      flush();
    }
  }
  if(groups.empty()){r.reason="no MMAC groups";return r;}
  auto *priority=Intrinsic::getOrInsertDeclaration(&m,Intrinsic::amdgcn_s_setprio);
  auto *fence=Intrinsic::getOrInsertDeclaration(&m,Intrinsic::amdgcn_sched_barrier);
  for(auto &g:groups) {
    // mode1 is same-instruction control: both writes use priority0. mode2
    // matches the native high scope; mode3 leaves the first MMAC low, as in
    // the HCU dot lowering's documented incoming-wave yield policy.
    if(mode==3 && g.size()<2)continue;
    Instruction *entry=mode==3?g.front()->getNextNode():g.front();
    IRBuilder<> before(entry);auto *hi=before.CreateCall(priority,{before.getInt16(mode==1?0:2)});hi->setConvergent();
    before.CreateCall(fence,{before.getInt32(0)});
    IRBuilder<> after(g.back()->getNextNode());after.CreateCall(fence,{after.getInt32(0)});
    auto *lo=after.CreateCall(priority,{after.getInt16(0)});lo->setConvergent();
    ++r.groups;r.mmas+=g.size();
  }
  std::string error;raw_string_ostream os(error);
  if(verifyModule(m,&os))throw std::runtime_error("MMAC priority produced invalid LLVM: "+error);
  r.reason="scoped priority, reset before memory/control exits; MMAC order unchanged";return r;
}
} // namespace hcu_tle_priority
