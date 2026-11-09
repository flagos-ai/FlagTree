#include "SharedAddressEvaluator.h"
#include "MemoryDrainSuffix.h"
#include "RegisterIdentityFence.h"
#include "TritonHCUGPUToLLVM/PostMmaWait.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include <deque>
#include <iterator>
#include <optional>
#include <vector>
using namespace llvm;
namespace flagtree::hcu {
namespace {
StringRef callee(Instruction *i) {
  auto *c=dyn_cast<CallBase>(i);auto *f=c?c->getCalledFunction():nullptr;
  return f?f->getName():StringRef();
}
bool named(Instruction *i,StringRef s) {return callee(i).starts_with(s);}
bool copy(Instruction *i) {return named(i,"llvm.amdgcn.raw.ptr.buffer.load.lds") || named(i,"llvm.amdgcn.struct.ptr.buffer.load.lds");}
bool matrix(Instruction *i) {return named(i,"llvm.hcu.ds.read.m32x16.b16.");}
bool mma(Instruction *i) {auto n=callee(i);return n.starts_with("llvm.hcu.mmac.") || n.starts_with("llvm.amdgcn.mfma.");}
InlineAsm *assembly(Instruction *i) {auto *c=dyn_cast<CallBase>(i);return c?dyn_cast<InlineAsm>(c->getCalledOperand()):nullptr;}
bool barrier(Instruction *i) {
  auto *a=assembly(i);return named(i,"llvm.amdgcn.s.barrier") || (a && StringRef(a->getAsmString()).trim()=="s_barrier");
}
bool fenceNoop(Instruction *i) {
  auto *a=assembly(i);return named(i,"llvm.amdgcn.sched.barrier") || (a && i->getType()->isVoidTy() && StringRef(a->getAsmString()).trim().empty());
}
struct Wait {unsigned vm,lgkm;};
std::optional<Wait> wait(Instruction *i) {
  if(!named(i,"llvm.amdgcn.s.waitcnt"))return std::nullopt;
  auto *c=cast<CallBase>(i);auto *k=dyn_cast<ConstantInt>(c->getArgOperand(0));if(!k)return std::nullopt;
  uint64_t x=k->getZExtValue();if(((x>>4)&7)!=7)return std::nullopt;
  return Wait{unsigned((x&15)|(((x>>14)&3)<<4)),unsigned((x>>8)&15)};
}
bool fenceInst(Instruction *i) {return barrier(i)||fenceNoop(i)||bool(wait(i));}
bool uniform(Value *v,SmallPtrSetImpl<Value *> &active) {
  if(isa<Constant>(v))return true;
  if(auto *a=dyn_cast<Argument>(v))return a->getType()->isIntegerTy() && a->hasInRegAttr();
  if(!active.insert(v).second)return true;
  auto *i=dyn_cast<Instruction>(v);bool ok=false;
  if(i && named(i,"llvm.amdgcn.workgroup.id."))ok=true;
  else if(i && (isa<BinaryOperator,CastInst,ICmpInst,SelectInst,PHINode>(i) || named(i,"llvm.smin.") || named(i,"llvm.smax.") || named(i,"llvm.umin.") || named(i,"llvm.umax."))) {
    ok=true;for(Value *x:i->operands())if(!isa<BasicBlock>(x))ok &= uniform(x,active);
  }
  active.erase(v);return ok;
}
using Eval = memory_schedule::SharedAddressEvaluator;
enum Kind {Noop,Copy,Read,VMWait,LDSZero,CTA};
struct Event {Kind kind;uint64_t mask=0;unsigned count=0;};
struct Events {
  Eval eval;unsigned sharedBytes;
  std::optional<Event> operator()(Instruction *i) {
    if(auto w=wait(i)) {
      if(w->vm<63 && w->lgkm<15)return std::nullopt; // keep v1 exact split waits
      if(w->vm<63)return Event{VMWait,0,w->vm};
      if(w->lgkm==0)return Event{LDSZero};
      if(w->lgkm==15)return Event{Noop};
      return std::nullopt; // partial response/WAR proof is deliberately absent
    }
    if(named(i,"llvm.amdgcn.s.waitcnt"))return std::nullopt;
    if(barrier(i))return Event{CTA};
    if(fenceNoop(i))return Event{Noop};
    if(isRegisterIdentityFence(i))return Event{Noop};
    // A true assumption records facts but issues no memory/queue event.
    // Keep it and its SSA operands in place, as ready-read scheduling does.
    if (auto *assume = dyn_cast<AssumeInst>(i))
      if (auto *condition = dyn_cast<ConstantInt>(assume->getArgOperand(0));
          condition && condition->isOne())
        return Event{Noop};
    uint64_t mask=0;
    if(copy(i)) {
      auto *c=cast<CallBase>(i);auto *size=dyn_cast<ConstantInt>(c->getArgOperand(2));if(!size||size->getZExtValue()!=16)return std::nullopt;
      bool raw=named(i,"llvm.amdgcn.raw.");
      uint64_t occupied=0, waveBase=0;
      for(unsigned t=0;t<eval.threads;++t) {
        auto p=eval(c->getArgOperand(1),t);if(!p)return std::nullopt;
        uint64_t low=*p&65535,high=*p&~uint64_t(65535);unsigned pw=(low%8192)/1024;
        uint64_t expected=uint64_t(pw&(raw?1:3))<<(raw?18:16);
        // A complete16B wave covers the same1KiB region for ordinary
        // zero-control placement and the legacy within-region rotation.
        // Both maps have independent full LDS readback evidence; this model
        // needs their footprint, not the payload's source ownership.
        if(low%1024 || (high!=0 && high!=expected) || low+1024>sharedBytes)return std::nullopt;
        if(t%64==0) {
          waveBase=*p;uint64_t bit=uint64_t(1)<<(low/1024);
          if(occupied&bit)return std::nullopt;occupied|=bit;
        } else if(*p!=waveBase)return std::nullopt;
        mask|=uint64_t(1)<<(low/1024);
      }
      return Event{Copy,mask};
    }
    Value *pointer=nullptr;unsigned bytes=0;uint64_t imm=0;
    if(auto *l=dyn_cast<LoadInst>(i)) {
      if(l->getPointerAddressSpace()!=3 || l->isVolatile() || l->isAtomic())return std::nullopt;
      auto sz=eval.dl.getTypeStoreSize(l->getType());if(sz.isScalable())return std::nullopt;
      bytes=sz.getFixedValue();if(!bytes||bytes>16)return std::nullopt;pointer=l->getPointerOperand();
    } else if(matrix(i)) {
      auto *c=cast<CallBase>(i);auto *k=dyn_cast<ConstantInt>(c->getArgOperand(1));if(!k||k->getZExtValue()>65535)return std::nullopt;
      pointer=c->getArgOperand(0);bytes=16;imm=k->getZExtValue();
    }
    if(pointer) {
      for(unsigned t=0;t<eval.threads;++t) {
        auto p=eval(pointer,t);if(!p)return std::nullopt;uint64_t lo=*p+imm;
        if(lo+bytes>sharedBytes || lo/8192!=(lo+bytes-1)/8192)return std::nullopt;
        // The normal matrix gather selects a source lane in the same wave
        // and adds 2*(outputLane&7). Union over ALL lane pointers plus 16B
        // therefore contains every implicit read. Ordinary loads use their
        // actual byte width. Keep a conservative 1KB physical-byte cover, rather
        // than invent reads of unwritten padding in a partial role region.
        unsigned first=lo/1024,last=(lo+bytes-1)/1024;
        for(unsigned chunk=first;chunk<=last;++chunk)
          mask|=uint64_t(1)<<chunk;
      }
      return Event{Read,mask};
    }
    if(isa<StoreInst,AtomicRMWInst,AtomicCmpXchgInst,FenceInst>(i))return std::nullopt;
    if(auto *c=dyn_cast<CallBase>(i)) {
      if(mma(i)) {if(c->mayReadOrWriteMemory() || c->mayHaveSideEffects())return std::nullopt;return Event{Noop};}
      if(auto *a=assembly(i)) {if(!a->hasSideEffects() && !StringRef(a->getConstraintString()).contains("memory"))return Event{Noop};return std::nullopt;}
      if(c->mayReadOrWriteMemory() || c->mayHaveSideEffects())return std::nullopt;
    }
    return Event{Noop};
  }
};
struct State {
  std::deque<uint64_t> dma;uint64_t ready=0,completed=0,reads=0,retiredReads=0;
  bool operator==(const State &x)const{return dma==x.dma&&ready==x.ready&&completed==x.completed&&reads==x.reads&&retiredReads==x.retiredReads;}
  bool step(Event e,bool allowPendingCTA=false) {
    if(e.kind==Copy) {
      if((reads|(allowPendingCTA?retiredReads:0))&e.mask)return false;
      for(uint64_t p:dma)if(p&e.mask)return false;
      ready&=~e.mask;dma.push_back(e.mask);
    } else if(e.kind==Read) {
      if((ready&e.mask)!=e.mask)return false;
      for(uint64_t p:dma)if(p&e.mask)return false;
      reads|=e.mask;
    } else if(e.kind==VMWait) {
      while(dma.size()>e.count){completed|=dma.front();dma.pop_front();}
    } else if(e.kind==LDSZero) {
      // A wait completes this wave only. Until the next common CTA, another
      // wave can still be using these addresses while a fast wave advances.
      if(allowPendingCTA)retiredReads|=reads;reads=0;
    }
    else if(e.kind==CTA) {
      // A barrier publishes completed DMA but does not complete unrelated LDS
      // reads. Pending reads may cross only if publication writes disjoint
      // physical regions. Copy and Read retain their original WAR/RAW checks.
      if(reads && (!allowPendingCTA || (completed&reads)))return false;
      ready|=completed;completed=0;retiredReads=0;
    }
    return true;
  }
};
struct Group {SmallVector<Instruction *> fence,copies;Instruction *mma=nullptr,*lastMma=nullptr;};
}
PostMmaWaitStats placePostMmaWaits(Module &m,const std::string &arch,const PostMmaWaitOptions &o) {
  PostMmaWaitStats s;
  std::string failedEvent;
  auto skip=[&](StringRef why){s.reason=why.str();if(!failedEvent.empty())s.reason+="; "+failedEvent;return s;};
  if(arch!="gfx936" || m.getDataLayout().getPointerSizeInBits(3)!=32)return skip("unsupported target/address ABI");
  if(!o.threads || o.threads%64 || o.threads>1024 || !o.sharedBytes || o.sharedBytes>65536 || o.sharedBytes%8192 || o.modelIterations<3)return skip("invalid exact ABI/domain options");
  GlobalVariable *shared=nullptr;
  for(auto &g:m.globals())if(g.getAddressSpace()==3) {if(shared||!g.isDeclaration())return skip("unknown AS3 global origin");shared=&g;}
  if(!shared)return skip("no single external AS3 origin");
  Function *kernel=nullptr;
  for(auto &f:m)if(!f.isDeclaration()) {if(kernel)return skip("multiple defined functions");kernel=&f;}
  if(!kernel || kernel->getCallingConv()!=CallingConv::AMDGPU_KERNEL)return skip("no single AMDGPU kernel");
  DominatorTree dt(*kernel);LoopInfo li(dt);auto loops=li.getLoopsInPreorder();
  if(loops.size()!=1)return skip("not exactly one loop");
  Loop *loop=loops[0];s.considered=1;
  if(loop->getNumBlocks()!=1 || !loop->getSubLoops().empty())return skip("not one self-loop block");
  BasicBlock *body=loop->getHeader(),*pre=loop->getLoopPreheader();
  auto *enter=pre?dyn_cast<BranchInst>(pre->getTerminator()):nullptr;
  if(!enter || enter->isConditional() || pre!=&kernel->getEntryBlock())return skip("not one unconditional outside entry");
  auto *back=dyn_cast<BranchInst>(body->getTerminator());
  if(!back || !back->isConditional() || (back->getSuccessor(0)!=body && back->getSuccessor(1)!=body))return skip("unsupported latch");
  for(auto &b:*kernel)if(auto *br=dyn_cast<BranchInst>(b.getTerminator()))if(br->isConditional()) {
    SmallPtrSet<Value *,32> active;if(!uniform(br->getCondition(),active))return skip("nonuniform function control");
  }
  Events event{{m.getDataLayout(),shared,o.threads},o.sharedBytes};
  SmallVector<Instruction *> ins;for(auto &i:*body)if(!isa<PHINode>(i))ins.push_back(&i);
  SmallVector<Group> groups;
  for(auto *i:ins)if(barrier(i)) {
    Group g;Instruction *first=i,*last=i;
    while(first->getPrevNode() && fenceInst(first->getPrevNode()))first=first->getPrevNode();
    while(last->getNextNode() && fenceNoop(last->getNextNode()))last=last->getNextNode();
    bool vm=false,lgkm=false;
    for(auto *j=first;;j=j->getNextNode()) {g.fence.push_back(j);if(auto w=wait(j)){vm|=w->vm<63;lgkm|=w->lgkm==0;if(w->lgkm!=0&&w->lgkm!=15)return skip("already partial/rotated");}if(j==last)break;}
    if(!vm||!lgkm)return skip("missing exact original VM/fullLGKM fence");
    for(auto *j:g.fence)if(!j->getType()->isVoidTy())return skip("nonvoid fence SSA dependency");
    groups.push_back(g);
  }
  if(groups.empty())return skip("no publication groups");
  for(unsigned n=0;n<groups.size();++n) {
    Instruction *begin=groups[n].fence.back()->getNextNode();Instruction *end=n+1<groups.size()?groups[n+1].fence.front():body->getTerminator();
    unsigned nr=0,nc=0,nm=0;bool seenMma=false;
    for(auto *i=begin;i!=end;i=i->getNextNode()) {
      if(!i)return skip("group does not close");
      auto e=event(i);if(!e)return skip("unknown group memory/event");
      if(mma(i)){if(!seenMma)groups[n].mma=i;groups[n].lastMma=i;seenMma=true;++nm;}
      if(e->kind==Read){if(seenMma)return skip("interleaved LDS after first MMA");++nr;}
      if(e->kind==Copy){if(seenMma)return skip("copy after first MMA");groups[n].copies.push_back(i);++nc;}
      if(e->kind==VMWait||e->kind==LDSZero||e->kind==CTA)return skip("extra queue fence in group");
    }
    if(!nr||!nc||!nm)return skip("not readNext/copyFuture/currentMMA group");
    // Moving M newest DMA issues past this VM wait preserves its original
    // FIFO completion frontier exactly when C>=M: wait(C) after append(M)
    // equals wait(C-M) before append(M). Never derive M from shape/stages.
    for(auto *i:groups[n].fence)if(auto w=wait(i))
      if(w->vm<63 && w->vm<groups[n].copies.size())
        return skip("VM threshold cannot retain moved DMA copies");
  }
  // A preexisting rotated body has a read/copy before its first fence. Never
  // rotate that source again, even if later waits happen to be full-zero.
  for(auto *i=&body->front();i!=groups[0].fence.front();i=i->getNextNode()) {
    auto e=event(i);if(!e||e->kind!=Noop)return skip("memory before first fence: already rotated or unsupported");
  }
  SmallPtrSet<Instruction *,32> removed;
  for(auto &g:groups){for(auto *i:g.fence)removed.insert(i);for(auto *i:g.copies)removed.insert(i);}
  std::vector<Event> prefix,prime,rotated;
  for(auto &i:*pre) {if(i.isTerminator())break;auto e=event(&i);if(!e)return skip("unknown preheader memory/event");prefix.push_back(*e);}
  for(auto *i:groups[0].fence){auto e=event(i);if(!e)return skip("unknown prime event");prime.push_back(*e);}
  for(auto &i:*body) {
    for(auto &g:groups)if(&i==g.mma) {
      for(auto *j:g.fence){auto e=event(j);if(!e)return skip("unknown relocated fence");
        if(e->kind==VMWait)e->count-=g.copies.size();
        if(e->kind!=LDSZero)rotated.push_back(*e);
      }
      // The common CTA has now discharged other waves' retired-read epoch.
      for(auto *j:g.copies){auto e=event(j);if(!e)return skip("unknown relocated copy");rotated.push_back(*e);}
    }
    if(!removed.contains(&i)){auto e=event(&i);if(!e)return skip("unknown loop event");rotated.push_back(*e);}
    for(auto &g:groups)if(&i==g.lastMma)for(auto *j:g.fence)if(auto w=wait(j))if(w->lgkm==0)rotated.push_back(Event{LDSZero});
  }
  State state;auto run=[&](const std::vector<Event> &events,bool allowPendingCTA=false){for(auto e:events){++s.modelEvents;if(!state.step(e,allowPendingCTA)){failedEvent="event="+std::to_string(unsigned(e.kind))+",mask="+std::to_string(e.mask)+",reads="+std::to_string(state.reads)+",retiredReads="+std::to_string(state.retiredReads);return false;}}return true;};
  std::vector<Event> original;
  for(auto &i:*body){auto e=event(&i);if(!e)return skip("unknown original loop event");original.push_back(*e);}
  if(!run(prefix))return skip("original prefix FIFO/publication proof failed");
  State originalFixed=state;
  for(unsigned k=0;k<o.modelIterations;++k) {
    if(!run(original)||!(state==originalFixed))return skip("original ring fixed-point proof failed");
  }
  state=State{};
  if(!run(prefix)||!run(prime))return skip("prime FIFO/publication/read WAR proof failed");
  // The prime has no retired reads; one mandatory first iteration reaches
  // the loop-cut state containing the last group's cross-wave read epoch.
  if(!run(rotated,true))return skip("first-iteration read/DMA WAR or publication proof failed");
  State fixed=state;
  for(unsigned k=1;k<o.modelIterations;++k) {if(!run(rotated,true))return skip("ring read/DMA WAR or publication proof failed");if(!(state==fixed))return skip("ring state does not return to post-wait fixed point");}
  auto *exit=loop->getUniqueExitBlock();
  if(!exit)return skip("no unique unchanged drain");
  // Every additional block must be an acyclic global-store-only suffix.
  // Its entry is the unique unchanged drain; no LDS event is accepted there.
  if(!pureGlobalStoreSuffix(*kernel,pre,body,exit,[&](Instruction *i) {
       auto e=event(i);return e && e->kind==Noop;
     }))return skip("unmodeled suffix CFG or shared/queue effect");
  for(auto &i:*exit) {
    if(copy(&i))return skip("unmodeled drain DMA");
    // Output is not shared storage. All source output and register SSA remain
    // in order; native counter readiness is a separate required check.
    if(auto *st=dyn_cast<StoreInst>(&i)) {if(st->getPointerAddressSpace()!=1)return skip("unknown drain store");continue;}
    if(named(&i,"llvm.amdgcn.raw.ptr.buffer.store") || named(&i,"llvm.amdgcn.struct.ptr.buffer.store"))continue;
    auto e=event(&i);if(!e || !state.step(*e))return skip("unchanged drain read/publication proof failed");++s.modelEvents;
  }
  if(!state.dma.empty())return skip("drain leaves pending DMA");
  // Exit/drain is unchanged. Any effect not modeled here may remain after the
  // final barrier; all pending writes are already non-overlapping and unchanged.
#if LLVM_VERSION_MAJOR >= 22
  for(auto *i:groups[0].fence)i->clone()->insertBefore(pre->getTerminator()->getIterator());
#else
  for(auto *i:groups[0].fence)i->clone()->insertBefore(pre->getTerminator());
#endif
  for(auto &g:groups) {
    Instruction *post=g.lastMma;
    for(auto *i:g.fence) {
      auto w=wait(i);bool after=w && w->lgkm==0;
#if LLVM_VERSION_MAJOR >= 22
      if(after)i->moveBefore(std::next(post->getIterator()));else i->moveBefore(g.mma->getIterator());
#else
      if(after)i->moveAfter(post);else i->moveBefore(g.mma);
#endif
      if(after)post=i;
      if(w && w->vm<63) {
        auto *call=cast<CallBase>(i);auto *old=cast<ConstantInt>(call->getArgOperand(0));
        uint64_t count=w->vm-g.copies.size();
        uint64_t encoded=(old->getZExtValue()&~uint64_t(0xc00f))|
                         (count&15)|((count&48)<<10);
        call->setArgOperand(0,ConstantInt::get(old->getType(),encoded));
        ++s.tightenedVMWaits;
      }
    }
    // All descriptor/address operands dominate the original earlier void
    // call; moving only the copy later in this same block preserves SSA.
    for(auto *copy:g.copies) {
#if LLVM_VERSION_MAJOR >= 22
      copy->moveBefore(g.mma->getIterator());
#else
      copy->moveBefore(g.mma);
#endif
      ++s.movedCopies;
    }
  }
  // A marker requests mandatory native validation on imported/cached IR; it
  // is never an input proof or a reason to bypass the structural checks.
  kernel->addFnAttr("hcu.memory.post_mma_wait","post-mma-lgkm0-copy-after-cta");
  s.transformed=1;s.groups=groups.size();s.reason="PASS_POST_MMA_FULL_LGKM0_COPY_AFTER_CTA_AND_CROSS_WAVE_FIXED_POINT";return s;
}
}
