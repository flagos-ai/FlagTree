// Private TLE gfx938 panel pipeline prototype. No floating point reassociation.
// This first version drains each source iteration: it never speculates a global
// address from a later iteration or assumes allocation padding.
#pragma once
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <optional>
#include <functional>
#include <stdexcept>

namespace hcu_tle_panel {
using namespace llvm;
struct Report {
  unsigned loops = 0, panels = 0, shared = 0;
  unsigned streamed = 0, rescaled = 0;
  unsigned crossed = 0, compacted = 0, unrolled = 0, deferred = 0;
  std::string reason;
};
inline StringRef name(CallBase *c) {
  auto *f = c->getCalledFunction();
  return f ? f->getName() : StringRef();
}
inline bool isMLS(CallBase *c) { return name(c).starts_with("llvm.hcu.matrix.load."); }
inline bool isDS(CallBase *c) { return name(c).starts_with("llvm.hcu.ds.read.matrix."); }
inline bool isMMA(CallBase *c) { return name(c).starts_with("llvm.hcu.mmac."); }
inline bool constant(Value *v, uint64_t k) {
  auto *c = dyn_cast<ConstantInt>(v);
  return c && c->getZExtValue() == k;
}
inline bool pureStateOp(Instruction *i) {
  if(isa<BinaryOperator>(i) || isa<CastInst>(i) || isa<InsertElementInst>(i) ||
     isa<ExtractElementInst>(i) || isa<ShuffleVectorInst>(i) || isa<SelectInst>(i) || isa<CmpInst>(i))return true;
  auto *c=dyn_cast<CallInst>(i);if(!c || c->mayReadOrWriteMemory())return false;
  auto n=name(c);
  return n=="llvm.maxnum.f32" || n=="llvm.minnum.f32" || n=="llvm.amdgcn.exp2.f32" || n.starts_with("llvm.fma.");
}
// Bisimulation is anchored at matching constant loop-entry values. The only
// recursive edge is a pair of PHIs in the same block; each backedge must then
// have precisely the same pure expression, flags, and non-recursive operands.
// This removes duplicated online-softmax state without changing any FP tree.
inline bool sameStateOperation(Instruction *x,Instruction *y) {
  if(!x->isSameOperationAs(y))return false;
  // isSameOperationAs can ignore flags whose violation creates poison.
  // Compare exact operations after temporarily matching their SSA operands.
  auto *probe=x->clone();
  for(unsigned k=0;k<probe->getNumOperands();++k)probe->setOperand(k,y->getOperand(k));
  bool same=probe->isIdenticalTo(y);probe->deleteValue();return same;
}
inline bool equalState(Value *a,Value *b,BasicBlock *bb,
                       DenseSet<std::pair<Value *,Value *>> &seen) {
  if(a==b)return true;
  if(a->getType()!=b->getType())return false;
  auto *x=dyn_cast<Instruction>(a),*y=dyn_cast<Instruction>(b);
  if(!x || !y || x->getParent()!=bb || y->getParent()!=bb)return false;
  if(!sameStateOperation(x,y))return false;
  if(auto *p=dyn_cast<PHINode>(x)) {
    auto *q=cast<PHINode>(y);
    if(p->getNumIncomingValues()!=2 || !p->getType()->isFloatTy())return false;
    int entry=-1,back=-1;
    for(unsigned k=0;k<2;++k) {
      if(p->getIncomingBlock(k)!=q->getIncomingBlock(k))return false;
      if(p->getIncomingBlock(k)==bb)back=k;else entry=k;
    }
    if(entry<0 || back<0 || !isa<ConstantFP>(p->getIncomingValue(entry)) ||
       p->getIncomingValue(entry)!=q->getIncomingValue(entry))return false;
    if(!seen.insert({a,b}).second)return true;
    return equalState(p->getIncomingValue(back),q->getIncomingValue(back),bb,seen);
  }
  if(!pureStateOp(x) || !pureStateOp(y))return false;
  if(!seen.insert({a,b}).second)return true;
  for(unsigned k=0;k<x->getNumOperands();++k)if(!equalState(x->getOperand(k),y->getOperand(k),bb,seen))return false;
  return true;
}
inline unsigned foldEquivalentState(Module &m) {
  unsigned count=0;
  for(auto &f:m)if(f.getCallingConv()==CallingConv::AMDGPU_KERNEL)for(auto &bb:f) {
    SmallVector<PHINode *> representatives,dead;
    for(auto &p:bb.phis()) {
      bool matched=false;
      for(auto *q:representatives) {
        DenseSet<std::pair<Value *,Value *>> seen;
        if(equalState(&p,q,&bb,seen)){p.replaceAllUsesWith(q);dead.push_back(&p);++count;matched=true;break;}
      }
      if(!matched)representatives.push_back(&p);
    }
    for(auto *p:dead)p->eraseFromParent();
    // A local pure CSE exposes the identical max/alpha chain immediately to
    // the following panel pass. Memory, synchronization and MMAC are excluded.
    DenseMap<size_t,SmallVector<Instruction *,2>> available;
    SmallVector<Instruction *> redundant;
    for(auto &i:bb)if(pureStateOp(&i)) {
      auto key=hash_combine(i.getOpcode(),i.getType());
      for(Value *op:i.operands())key=hash_combine(key,op);
      auto &bucket=available[size_t(key)];bool found=false;
      for(auto *old:bucket)if(i.isIdenticalTo(old)) {
        i.replaceAllUsesWith(old);redundant.push_back(&i);found=true;break;
      }
      if(!found)bucket.push_back(&i);
    }
    for(auto *i:redundant)i->eraseFromParent();
  }
  std::string errors;raw_string_ostream os(errors);
  if(verifyModule(m,&os))throw std::runtime_error("HCU loop-state fold produced invalid IR: "+errors);
  return count;
}
// Evaluate only the physical LDS address language, for every participating lane.
// AS3 base is deliberately kept symbolic as zero; tagged bit31 is removed only
// at the MLS destination check, never when evaluating a DS read.
inline std::optional<uint64_t> eval(Value *v, unsigned tid, const DataLayout &dl,
                                   unsigned depth = 0) {
  if (depth > 64) return {};
  if (auto *c = dyn_cast<ConstantInt>(v)) return c->getZExtValue();
  if (auto *g = dyn_cast<GlobalVariable>(v)) {
    if (g->getAddressSpace() == 3 && g->isDeclaration() &&
        isa<ArrayType>(g->getValueType()) &&
        cast<ArrayType>(g->getValueType())->getNumElements() == 0) return 0;
    return {};
  }
  if (auto *c = dyn_cast<CallBase>(v)) {
    if (name(c) == "llvm.amdgcn.workitem.id.x") return tid;
    if (name(c) == "llvm.amdgcn.readfirstlane.i32")
      return eval(c->getArgOperand(0), tid & ~63u, dl, depth + 1);
    return {};
  }
  auto *u = dyn_cast<Operator>(v);
  if (!u) return {};
  if (auto *g = dyn_cast<GEPOperator>(u)) {
    auto b = eval(g->getPointerOperand(), tid, dl, depth + 1);
    if (!b || g->getNumIndices() != 1 || !g->getSourceElementType()->isIntegerTy()) return {};
    auto n = eval(g->idx_begin()->get(), tid, dl, depth + 1);
    if (!n) return {};
    return *b + *n * dl.getTypeAllocSize(g->getSourceElementType()).getFixedValue();
  }
  auto a = eval(u->getOperand(0), tid, dl, depth + 1);
  if (!a) return {};
  uint64_t r;
  switch (u->getOpcode()) {
  case Instruction::PtrToInt: case Instruction::IntToPtr:
  case Instruction::BitCast: case Instruction::ZExt: r = *a; break;
  case Instruction::Trunc: r = *a; break;
  default: {
    if (u->getNumOperands() != 2) return {};
    auto b = eval(u->getOperand(1), tid, dl, depth + 1);
    if (!b) return {};
    switch (u->getOpcode()) {
    case Instruction::Add: r = *a + *b; break;
    case Instruction::Mul: r = *a * *b; break;
    case Instruction::And: r = *a & *b; break;
    case Instruction::Or: r = *a | *b; break;
    case Instruction::Xor: r = *a ^ *b; break;
    case Instruction::Shl: if (*b >= 64) return {}; r = *a << *b; break;
    case Instruction::LShr: if (*b >= 64) return {}; r = *a >> *b; break;
    default: return {};
    }
  }
  }
  if (u->getType()->isIntegerTy() && u->getType()->getIntegerBitWidth() < 64)
    r &= (uint64_t(1) << u->getType()->getIntegerBitWidth()) - 1;
  return r;
}
struct Panel {
  SmallVector<CallInst *> copies, reads, mmas;
  unsigned compactStride = 0;
};
struct Candidate {
  BasicBlock *bb = nullptr;
  SmallVector<Panel> panels;
  SmallVector<CallInst *> sync;
};
inline bool canClone(Value *v, BasicBlock *bb, SmallPtrSetImpl<Value *> &seen) {
  auto *i = dyn_cast<Instruction>(v);
  if (!i || i->getParent() != bb || isa<PHINode>(i)) return true;
  if (!seen.insert(v).second) return true;
  if (auto *c = dyn_cast<CallInst>(i)) {
    if (name(c) != "llvm.amdgcn.readfirstlane.i32") return false;
  } else if (!isa<BinaryOperator>(i) && !isa<CastInst>(i) &&
             !isa<GetElementPtrInst>(i) && !isa<InsertElementInst>(i) &&
             !isa<ExtractElementInst>(i) && !isa<ShuffleVectorInst>(i) &&
             !isa<SelectInst>(i) && !isa<ICmpInst>(i)) return false;
  for (Value *op : i->operands()) if (!canClone(op, bb, seen)) return false;
  return true;
}
inline bool rootedIn(Value *v, const Panel &p, SmallPtrSetImpl<Value *> &seen) {
  if (!seen.insert(v).second) return true;
  if (auto *c = dyn_cast<CallInst>(v)) return llvm::is_contained(p.reads, c);
  if (isa<Constant>(v)) return true;
  auto *i = dyn_cast<Instruction>(v);
  if (!i || (!isa<BitCastInst>(i) && !isa<ShuffleVectorInst>(i) &&
             !isa<ExtractElementInst>(i) && !isa<InsertElementInst>(i))) return false;
  for (Value *op : i->operands()) if (!rootedIn(op, p, seen)) return false;
  return true;
}
inline std::optional<Candidate> analyze(BasicBlock &bb, std::string &why, bool straight=false) {
  Instruction *begin=nullptr,*end=nullptr;
  if(straight) {
    CallInst *first=nullptr,*last=nullptr;
    for(auto &i:bb)if(auto *call=dyn_cast<CallInst>(&i)) {
      if(!first && isMLS(call))first=call;
      if(isMMA(call))last=call;
    }
    if(!first || !last || last->comesBefore(first))return {};
    SmallVector<CallInst *,2> prefix;
    for(auto *i=first->getPrevNode();i && prefix.size()<2;i=i->getPrevNode())
      if(auto *call=dyn_cast<CallInst>(i);call &&
         (name(call)=="llvm.amdgcn.s.waitcnt" || name(call)=="llvm.amdgcn.s.barrier"))prefix.push_back(call);
    if(prefix.size()!=2 || name(prefix[0])!="llvm.amdgcn.s.barrier" ||
       name(prefix[1])!="llvm.amdgcn.s.waitcnt" || !constant(prefix[1]->getArgOperand(0),49279))return {};
    begin=prefix[1];end=last->getNextNode();
  }
  Candidate c; c.bb = &bb;
  bool hasRead = false,active=!straight;
  for (Instruction &i : bb) {
    if(&i==begin)active=true;
    if(&i==end)break;
    if(!active)continue;
    if (isa<LoadInst>(i) || isa<StoreInst>(i) || isa<AtomicRMWInst>(i) ||
        isa<AtomicCmpXchgInst>(i) || isa<FenceInst>(i)) { why="ordinary memory in loop"; return {}; }
    auto *call = dyn_cast<CallInst>(&i);
    if (!call) continue;
    if (isMLS(call)) {
      if (c.panels.empty() || hasRead) { c.panels.emplace_back(); hasRead=false; }
      c.panels.back().copies.push_back(call);
    } else if (isDS(call)) {
      if (c.panels.empty()) { why="unpaired DS read"; return {}; }
      c.panels.back().reads.push_back(call); hasRead=true;
    } else if (isMMA(call)) {
      if (c.panels.empty()) { why="unpaired MMA"; return {}; }
      c.panels.back().mmas.push_back(call);
    } else if (name(call) == "llvm.amdgcn.s.waitcnt" || name(call) == "llvm.amdgcn.s.barrier") {
      c.sync.push_back(call);
    } else if (call->mayReadOrWriteMemory() || call->isInlineAsm()) {
      why="unknown memory effects"; return {};
    }
  }
  if (c.panels.size() < 2 || c.panels.size() % 2) { why="need even panel sequence"; return {}; }
  if (c.sync.size() != 5*c.panels.size()) { why="unexpected synchronization"; return {}; }
  auto &dl = bb.getModule()->getDataLayout();
  for (unsigned j=0; j<c.panels.size(); ++j) {
    auto &p=c.panels[j];
    if ((p.reads.size()!=4 && p.reads.size()!=8) || p.mmas.empty() || p.copies.empty() || p.copies.size()>2) { why="unsupported panel extent"; return {}; }
    // Exact source protocol: LDS WAR wait+barrier, copies, VM0+LDS0+RAW barrier.
    unsigned k=5*j;
    if (name(c.sync[k])!="llvm.amdgcn.s.waitcnt" || !constant(c.sync[k]->getArgOperand(0),49279) ||
        name(c.sync[k+1])!="llvm.amdgcn.s.barrier" ||
        name(c.sync[k+2])!="llvm.amdgcn.s.waitcnt" || !constant(c.sync[k+2]->getArgOperand(0),uint32_t(-49168)) ||
        name(c.sync[k+3])!="llvm.amdgcn.s.waitcnt" || !constant(c.sync[k+3]->getArgOperand(0),49279) ||
        name(c.sync[k+4])!="llvm.amdgcn.s.barrier" ||
        !c.sync[k+1]->comesBefore(p.copies.front()) ||
        !p.copies.back()->comesBefore(c.sync[k+2]) ||
        !c.sync[k+4]->comesBefore(p.reads.front())) { why="unsupported source wait protocol"; return {}; }
    for (auto *x:p.copies) {
      auto n=name(x);
      unsigned bytes=n=="llvm.hcu.matrix.load.32x16.b16.p3"?1024:
                     n=="llvm.hcu.matrix.load.64x16.b16.p3"?2048:0;
      if (!bytes || x->arg_size()!=8 ||
          (!constant(x->getArgOperand(4),0) && !constant(x->getArgOperand(4),1)) ||
          !constant(x->getArgOperand(5),0) || !constant(x->getArgOperand(6),0) || !constant(x->getArgOperand(7),0) ||
          !isa<ConstantInt>(x->getArgOperand(2)) || !isa<ConstantInt>(x->getArgOperand(3))) { why="unsupported MLS ABI"; return {}; }
      SmallPtrSet<Value *,32> seen;
      for (Value *v:x->args()) if (!canClone(v,&bb,seen)) { why="non-rematerializable MLS address"; return {}; }
      for (unsigned t=0;t<256;++t) {
        auto a=eval(x->getArgOperand(1),t,dl), lead=eval(x->getArgOperand(1),t&~63u,dl);
        if (!a || !lead || *a!=*lead || (bytes==1024 && !(*a&0x80000000u)) || ((*a&0x7fffffffu)+bytes)>8192) { why="unproved LDS destination"; return {}; }
      }
    }
    for (auto *x:p.reads) {
      auto *ty=dyn_cast<FixedVectorType>(x->getType());
      if (x->arg_size()!=6 || !ty || ty->getNumElements()!=8 ||
          (!ty->getElementType()->isBFloatTy() && !ty->getElementType()->isHalfTy()) ||
          !constant(x->getArgOperand(2),2) || !constant(x->getArgOperand(3),2) ||
          !constant(x->getArgOperand(4),1) || !constant(x->getArgOperand(5),0)) { why="unsupported DS fragment"; return {}; }
      for(unsigned tid=0;tid<256;++tid) {
        auto a=eval(x->getArgOperand(0),tid,dl), off=eval(x->getArgOperand(1),tid,dl);
        if (!a || !off || *a+*off+1024>8192) { why="unproved DS extent"; return {}; }
      }
    }
    for (auto *x:p.mmas) {
      SmallPtrSet<Value *,32> seen;
      if (!rootedIn(x->getArgOperand(1),p,seen)) { why="MMA fragment crosses panels"; return {}; }
    }
  }
  return c;
}
inline CallInst *assembly(IRBuilder<> &b, StringRef code, ArrayRef<Value *> args={}, StringRef extra="") {
  SmallVector<Type *> types;
  for(auto *x:args) types.push_back(x->getType());
  auto *ft=FunctionType::get(b.getVoidTy(),types,false);
  auto *a=InlineAsm::get(ft,code.str(),extra.str()+"~{memory}",true);
  auto *call=b.CreateCall(a,args); call->setConvergent(); call->setDoesNotThrow(); return call;
}
inline CallInst *fragmentRoot(Value *v, const Panel &p,
                              SmallPtrSetImpl<Instruction *> &support) {
  if(auto *call=dyn_cast<CallInst>(v))return llvm::is_contained(p.reads,call)?call:nullptr;
  auto *i=dyn_cast<Instruction>(v);
  if(!i || (!isa<BitCastInst>(i) && !isa<ShuffleVectorInst>(i)))return nullptr;
  CallInst *root=nullptr;
  for(Value *op:i->operands()) {
    if(isa<Constant>(op))continue;
    auto *r=fragmentRoot(op,p,support);
    if(!r || (root && root!=r))return nullptr;
    root=r;
  }
  if(root)support.insert(i);return root;
}
inline unsigned streamV(Candidate &c) {
  unsigned changed=0;
  for(auto &p:c.panels) {
    if(p.copies.size()!=1 || name(p.copies[0])!="llvm.hcu.matrix.load.64x16.b16.p3" ||
       !constant(p.copies[0]->getArgOperand(3),1) || p.mmas.size()!=32)continue;
    SmallPtrSet<Instruction *,32> support;
    DenseMap<CallInst *,unsigned> first;
    bool valid=true;
    for(unsigned n=0;n<32;++n) {
      auto *root=fragmentRoot(p.mmas[n]->getArgOperand(1),p,support);
      if(!root){valid=false;break;}
      if(!first.count(root))first[root]=n;
    }
    if(!valid || first.size()!=8)continue;
    SmallVector<CallInst *,2> groups[4];
    for(auto *ds:p.reads)groups[first[ds]/8].push_back(ds);
    for(auto &g:groups)if(g.size()!=2)valid=false;
    for(auto *i:support)for(auto *u:i->users()) {
      auto *ins=dyn_cast<Instruction>(u);
      if(!ins || (!support.contains(ins) && !llvm::is_contained(p.mmas,dyn_cast<CallInst>(ins))))valid=false;
    }
    if(!valid)continue;
    SmallPtrSet<Instruction *,32> prepared;
    std::function<void(Value *,Instruction *)> prepare=[&](Value *v,Instruction *at) {
      auto *i=dyn_cast<Instruction>(v);
      if(!i || !support.contains(i) || !prepared.insert(i).second)return;
      for(Value *op:i->operands())prepare(op,at);
      i->moveBefore(at->getIterator());
    };
    SmallVector<CallInst *> order;
    for(unsigned g=0;g<4;++g) {
      auto *at=p.mmas[g*8];
      IRBuilder<> b(at);
      if(g) {
        auto fence=c.bb->getModule()->getOrInsertFunction("llvm.amdgcn.sched.barrier",FunctionType::get(b.getVoidTy(),{b.getInt32Ty()},false));
        b.CreateCall(fence,{b.getInt32(0)});
      }
      if(g==0)for(unsigned x=0;x<3;++x)for(auto *ds:groups[x]){ds->moveBefore(at->getIterator());order.push_back(ds);}
      if(g==1)for(auto *ds:groups[3]){ds->moveBefore(at->getIterator());order.push_back(ds);}
      for(unsigned n=g*8;n<g*8+8;++n)prepare(p.mmas[n]->getArgOperand(1),p.mmas[n]);
    }
    p.reads=std::move(order);++changed;
  }
  return changed;
}
inline bool phiPack(Value *v, BasicBlock *bb, SmallPtrSetImpl<Value *> &seen) {
  if(!seen.insert(v).second || isa<Constant>(v))return true;
  auto *i=dyn_cast<Instruction>(v);
  if(!i)return false;
  if(auto *p=dyn_cast<PHINode>(i))return p->getParent()==bb && p->getType()->isFloatTy();
  if(!isa<InsertElementInst>(i) && !isa<ShuffleVectorInst>(i))return false;
  for(Value *op:i->operands())if(!phiPack(op,bb,seen))return false;
  return true;
}
inline Value *alphaSplat(Value *v) {
  auto *s=dyn_cast<ShuffleVectorInst>(v);
  if(!s)return nullptr;
  for(int n:s->getShuffleMask())if(n!=0)return nullptr;
  auto *i=dyn_cast<InsertElementInst>(s->getOperand(0));
  if(!i || !constant(i->getOperand(2),0))return nullptr;
  auto *a=dyn_cast<CallInst>(i->getOperand(1));
  return a && name(a)=="llvm.amdgcn.exp2.f32"?a:nullptr;
}
inline unsigned guardRescale(Candidate &c) {
  auto *bb=c.bb;auto *f=bb->getParent();
  if(f->getFnAttribute("denormal-fp-math-f32").getValueAsString()!="ieee")return 0;
  SmallVector<BinaryOperator *> muls;
  SmallVector<Value *> alphas;
  DenseMap<BinaryOperator *,Value *> original;
  for(auto &p:c.panels)for(auto *mma:p.mmas) {
    auto *x=dyn_cast<BinaryOperator>(mma->getArgOperand(2));
    if(!x || x->getOpcode()!=Instruction::FMul || x->getParent()!=bb || llvm::is_contained(muls,x))continue;
    auto *a=alphaSplat(x->getOperand(1));
    SmallPtrSet<Value *,32> seen;
    if(!a || !phiPack(x->getOperand(0),bb,seen))continue;
    muls.push_back(x);original[x]=x->getOperand(0);
    if(!llvm::is_contained(alphas,a))alphas.push_back(a);
  }
  if(muls.empty())return 0;
  auto *anchor=muls.front();
  for(auto *x:muls)if(x->comesBefore(anchor))anchor=x;
  for(auto *a:alphas)if(cast<Instruction>(a)->getParent()!=bb || !cast<Instruction>(a)->comesBefore(anchor))return 0;
  // Only pure packing can move before the branch. All real arithmetic stays.
  SmallPtrSet<Instruction *,32> moved;
  std::function<void(Value *)> hoist=[&](Value *v){
    auto *i=dyn_cast<Instruction>(v);
    if(!i || i->getParent()!=bb || isa<PHINode>(i) || i->comesBefore(anchor) || !moved.insert(i).second)return;
    for(Value *op:i->operands())hoist(op);
    i->moveBefore(anchor->getIterator());
  };
  for(auto *x:muls)for(Value *op:x->operands())hoist(op);
  auto *join=SplitBlock(bb,anchor);
  auto *scale=BasicBlock::Create(bb->getContext(),"panel.rescale",f,join);
  IRBuilder<> s(scale);auto *end=s.CreateBr(join);
  for(auto *x:muls)x->moveBefore(end->getIterator());
  bb->getTerminator()->eraseFromParent();IRBuilder<> b(bb);
  Value *need=b.getFalse();
  for(auto *a:alphas)need=b.CreateOr(need,b.CreateFCmpUNE(a,ConstantFP::get(a->getType(),1.)));
  auto vote=bb->getModule()->getOrInsertFunction("llvm.amdgcn.ballot.i64",FunctionType::get(b.getInt64Ty(),{b.getInt1Ty()},false));
  auto *mask=b.CreateCall(vote,{need});mask->setConvergent();
  b.CreateCondBr(b.CreateICmpNE(mask,b.getInt64(0)),scale,join);
  IRBuilder<> j(&*join->getFirstInsertionPt());
  for(auto *x:muls) {
    auto *phi=j.CreatePHI(x->getType(),2,"panel.identity");x->replaceAllUsesWith(phi);
    phi->addIncoming(original[x],bb);phi->addIncoming(x,scale);
  }
  return muls.size();
}
inline Value *cloneAddress(Value *v, BasicBlock *bb, IRBuilder<> &b, DenseMap<Value *,Value *> &map) {
  if(auto it=map.find(v);it!=map.end())return it->second;
  auto *i=dyn_cast<Instruction>(v);
  if(!i || i->getParent()!=bb || isa<PHINode>(i))return v;
  auto *n=i->clone();
  for(unsigned k=0;k<n->getNumOperands();++k)n->setOperand(k,cloneAddress(i->getOperand(k),bb,b,map));
  b.Insert(n,"panel.addr");map[v]=n;return n;
}
inline unsigned copies(const Panel &p) { return p.compactStride?1:p.copies.size(); }
inline void issue(IRBuilder<> &b, Candidate &c, unsigned j, ArrayRef<Value *> dependencies={},
                  Value *iv=nullptr, Value *iteration=nullptr) {
  DenseMap<Value *,Value *> map;
  if(iv)map[iv]=iteration;
  SmallVector<Value *> args(dependencies);
  std::string constraints;
  for (unsigned k=0;k<dependencies.size();++k) constraints+="v,";
  std::string code=dependencies.empty()?"":"s_waitcnt lgkmcnt(0)\ns_barrier\n";
  auto &p=c.panels[j];
  for(auto *x:p.copies) {
    if(p.compactStride && x!=p.copies.front())break;
    unsigned n=args.size();
    auto *rsrc=cloneAddress(x->getArgOperand(0),c.bb,b,map);
    auto *ptr=cloneAddress(x->getArgOperand(1),c.bb,b,map);
    auto *dst=b.CreatePtrToInt(ptr,b.getInt32Ty());
    if(p.compactStride) {
      auto tid=b.GetInsertBlock()->getModule()->getOrInsertFunction("llvm.amdgcn.workitem.id.x",FunctionType::get(b.getInt32Ty(),{},false));
      auto first=b.GetInsertBlock()->getModule()->getOrInsertFunction("llvm.amdgcn.readfirstlane.i32",FunctionType::get(b.getInt32Ty(),{b.getInt32Ty()},false));
      Value *wave=b.CreateLShr(b.CreateCall(tid),b.getInt32(6));wave=b.CreateCall(first,{wave});
      auto *wide=b.CreateBitCast(rsrc,FixedVectorType::get(b.getInt64Ty(),2));
      auto *address=b.CreateExtractElement(wide,b.getInt32(0));
      auto *delta=b.CreateZExt(b.CreateMul(wave,b.getInt32(32*p.compactStride)),b.getInt64Ty());
      wide=b.CreateInsertElement(wide,b.CreateAdd(address,delta),b.getInt32(0));
      rsrc=b.CreateBitCast(wide,rsrc->getType());
      dst=b.CreateAdd(dst,b.CreateShl(wave,b.getInt32(10)));
    }
    args.push_back(rsrc);
    if(j%2)dst=b.CreateAdd(dst,b.getInt32(8192));
    args.push_back(dst); constraints+="s,s,";
    std::string op=p.compactStride?"32x32":name(x).contains("32x16")?"32x16":"64x16";
    code+="s_nop 4\nmatrix_load_"+op+"_b16 $"+std::to_string(n)+" $"+std::to_string(n+1)+
          " moffset:"+std::to_string(cast<ConstantInt>(x->getArgOperand(2))->getZExtValue())+
          (constant(x->getArgOperand(3),1)?" t":"")+
          (constant(x->getArgOperand(4),1)?" r":"")+" lds\n";
  }
  assembly(b,code,args,constraints);
}
// All DS operands must already be issued before the first compute cluster.
// We keep their full LDS wait and CTA WAR barrier, but may execute MMAC8
// before that drain. Register consumers do not need to wait for unrelated
// later DS results; no slot can be overwritten before the original rendezvous.
struct ReadBundle {CallInst *read=nullptr;unsigned target=0;SmallVector<Instruction *> nodes;};
inline std::optional<SmallVector<ReadBundle>> staggerPlan(const Candidate &c,const Panel &p,unsigned after) {
  if((after!=16 && after!=24) || p.mmas.size()!=32 || p.reads.size()!=8)return {};
  SmallVector<ReadBundle> plans;SmallPtrSet<Instruction *,32> owned;
  unsigned lookahead=32-after;
  for(auto *read:p.reads) {
    ReadBundle plan;plan.read=read;SmallPtrSet<Instruction *,32> seen;SmallVector<Instruction *> work{read};unsigned first=32;
    while(!work.empty()) {
      auto *v=work.pop_back_val();if(!seen.insert(v).second)continue;
      plan.nodes.push_back(v);
      for(auto *u:v->users()) {
        auto *i=dyn_cast<Instruction>(u);if(!i || i->getParent()!=c.bb)return {};
        if(auto *call=dyn_cast<CallInst>(i)) {
          auto it=llvm::find(p.mmas,call);
          if(it!=p.mmas.end()){first=std::min(first,unsigned(it-p.mmas.begin()));continue;}
        }
        if(!isa<BitCastInst>(i) && !isa<ShuffleVectorInst>(i) && !isa<ExtractElementInst>(i) && !isa<InsertElementInst>(i))return {};
        work.push_back(i);
      }
    }
    if(first==32)return {};
    for(auto *i:plan.nodes) {
      if(!owned.insert(i).second)return {};
      if(i==read)continue;
      for(auto &op:i->operands()) {
        if(isa<Constant>(op))continue;
        auto *source=dyn_cast<Instruction>(op);
        if(!source || !seen.contains(source))return {};
      }
    }
    plan.target=first>lookahead?first-lookahead:0;
    if(plan.target>=after || !read->comesBefore(p.mmas[plan.target]))return {};
    llvm::sort(plan.nodes,[](auto *a,auto *b){return a->comesBefore(b);});plans.push_back(std::move(plan));
  }
  return plans;
}
inline void staggerReads(Candidate &c,Panel &p,unsigned after) {
  auto plans=staggerPlan(c,p,after);if(!plans)throw std::runtime_error("unproved stagger plan after preflight");
  SmallPtrSet<Instruction *,8> fenced;
  for(auto &plan:*plans) {
    auto *anchor=p.mmas[plan.target];
    if(fenced.insert(anchor).second) {
      IRBuilder<> b(anchor);auto f=c.bb->getModule()->getOrInsertFunction("llvm.amdgcn.sched.barrier",FunctionType::get(b.getVoidTy(),{b.getInt32Ty()},false));
      b.CreateCall(f,{b.getInt32(0)});
    }
    for(auto *i:plan.nodes)i->moveBefore(anchor->getIterator());
  }
}
inline bool deferEligible(const Candidate &c,unsigned after) {
  if(after!=8 && after!=16 && after!=24)return false;
  for(auto &p:c.panels) {
    if(p.reads.empty() || p.mmas.size()<after || !p.reads.back()->comesBefore(p.mmas[after-1]))return false;
    if(after>=16 && !staggerPlan(c,p,after))return false;
  }
  return true;
}
inline void transform(Candidate &c, bool prefetched=false, Value *iv=nullptr,
                      Value *nextStep=nullptr, bool nativeWar=false, unsigned warAfter=0) {
  auto &ctx=c.bb->getContext();
  auto *anchor=c.panels.front().copies.front(); IRBuilder<> pro(anchor);
  // Empty all pre-existing VM operations before counting outstanding MLS.
  if(!prefetched) {
    assembly(pro,"s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier");
    issue(pro,c,0); issue(pro,c,1);
  }
  for(unsigned j=0;j<c.panels.size();++j) {
    auto &p=c.panels[j]; IRBuilder<> raw(p.reads.front());
    unsigned young=j+1<c.panels.size()?copies(c.panels[j+1]):nextStep?copies(c.panels[0]):0;
    assembly(raw,"s_waitcnt vmcnt("+std::to_string(young)+") lgkmcnt(0)\ns_barrier");
    if(j%2) for(auto *ds:p.reads) {
      IRBuilder<> at(ds);
      ds->setArgOperand(0,at.CreateGEP(at.getInt8Ty(),ds->getArgOperand(0),at.getInt32(8192)));
    }
    if(warAfter>=16)staggerReads(c,p,warAfter);
    IRBuilder<> war(warAfter?p.mmas[warAfter-1]->getNextNode():p.reads.back()->getNextNode());
    SmallVector<Value *> deps;
    if(nativeWar) {
      // Both recognized intrinsics have ordering effects. All DS reads stay
      // before the LDS-completion wait and CTA WAR barrier; the next MLS asm
      // has an explicit memory clobber. Keep completion and reuse protection.
      auto wait=c.bb->getModule()->getOrInsertFunction("llvm.amdgcn.s.waitcnt",FunctionType::get(war.getVoidTy(),{war.getInt32Ty()},false));
      auto barrier=c.bb->getModule()->getOrInsertFunction("llvm.amdgcn.s.barrier",FunctionType::get(war.getVoidTy(),{},false));
      war.CreateCall(wait,{war.getInt32(49279)});war.CreateCall(barrier,{});
    } else for(auto *ds:p.reads)deps.push_back(war.CreateBitCast(ds,FixedVectorType::get(war.getInt32Ty(),4)));
    if(j+2<c.panels.size())issue(war,c,j+2,deps);
    else if(nextStep) {
      auto *add=cast<BinaryOperator>(nextStep);
      Value *next=war.CreateAdd(iv,add->getOperand(1));
      issue(war,c,j+2-c.panels.size(),deps,iv,next);
    }
    else if(!nativeWar) {
      std::string cs;for(auto *v:deps){(void)v;cs+="v,";}
      assembly(war,"s_waitcnt lgkmcnt(0)\ns_barrier",deps,cs);
    }
    // Limit overlapping fragment lifetimes without changing arithmetic order.
    auto *f=c.bb->getModule()->getOrInsertFunction("llvm.amdgcn.sched.barrier",FunctionType::get(Type::getVoidTy(ctx),{Type::getInt32Ty(ctx)},false)).getCallee();
    IRBuilder<> end(p.mmas.back()->getNextNode());
    end.CreateCall(cast<Function>(f),{end.getInt32(0)});
  }
  for(auto &p:c.panels)for(auto *x:p.copies)x->eraseFromParent();
  for(auto *x:c.sync)x->eraseFromParent();
}
// Only compact the previously byte-verified B16 layout: two32x16(r) copies,
// same resource, moffset0/64, four waves at1024B and+4096B, trans-format DS.
inline std::optional<uint64_t> lane(Value *v,unsigned n) {
  if(auto *i=dyn_cast<InsertElementInst>(v)) {
    auto *k=dyn_cast<ConstantInt>(i->getOperand(2));if(!k)return {};
    if(k->getZExtValue()==n) {
      auto *a=dyn_cast<ConstantInt>(i->getOperand(1));return a?std::optional<uint64_t>(a->getZExtValue()):std::nullopt;
    }
    return lane(i->getOperand(0),n);
  }
  if(auto *c=dyn_cast<Constant>(v))if(auto *e=dyn_cast_or_null<ConstantInt>(c->getAggregateElement(n)))return e->getZExtValue();
  return {};
}
inline unsigned compact(Candidate &c) {
  unsigned count=0;auto &dl=c.bb->getModule()->getDataLayout();
  for(auto &p:c.panels) {
    if(p.copies.size()!=2)continue;auto *a=p.copies[0],*b=p.copies[1];
    if(name(a)!="llvm.hcu.matrix.load.32x16.b16.p3" || name(b)!=name(a) ||
       a->getArgOperand(0)!=b->getArgOperand(0) || !constant(a->getArgOperand(2),0) ||
       !constant(b->getArgOperand(2),64) || !constant(a->getArgOperand(3),0) || !constant(b->getArgOperand(3),0))continue;
    auto stride=lane(a->getArgOperand(0),2),bound=lane(a->getArgOperand(0),3);
    if(!stride || !bound || *bound!=0 || *stride<128 || *stride>65535 || *stride%8)continue;
    bool valid=true;
    for(unsigned t=0;t<256;++t) {
      auto x=eval(a->getArgOperand(1),t,dl),y=eval(b->getArgOperand(1),t,dl);
      if(!x || !y || *x!=(0x80000000u+(t/64)*1024) || *y!=(*x+4096))valid=false;
    }
    for(auto *ds:p.reads)if(!name(ds).contains("matrix.trans.format"))valid=false;
    if(valid){p.compactStride=*stride;++count;}
  }
  return count;
}
struct Trip {
  PHINode *iv; BinaryOperator *step; ConstantInt *start;
  ICmpInst *cmp; BranchInst *br; BasicBlock *pre,*exit;
  uint64_t count;
};
inline std::optional<Trip> proveTrip(Candidate &c) {
  auto *bb=c.bb;auto *br=dyn_cast<BranchInst>(bb->getTerminator());
  if(!br || !br->isConditional() || br->getSuccessor(0)!=bb)return {};
  auto *cmp=dyn_cast<ICmpInst>(br->getCondition());
  if(!cmp || cmp->getPredicate()!=ICmpInst::ICMP_ULT)return {};
  auto *iv=dyn_cast<PHINode>(cmp->getOperand(0));
  auto *bound=dyn_cast<ConstantInt>(cmp->getOperand(1));
  if(!iv || !iv->getType()->isIntegerTy(32) || iv->getParent()!=bb || iv->getNumIncomingValues()!=2 || !bound)return {};
  auto *pre=iv->getIncomingBlock(0)==bb?iv->getIncomingBlock(1):iv->getIncomingBlock(0);
  auto *pb=dyn_cast<BranchInst>(pre->getTerminator());
  if(!pb || !pb->isUnconditional() || pb->getSuccessor(0)!=bb)return {};
  auto *start=dyn_cast<ConstantInt>(iv->getIncomingValueForBlock(pre));
  auto *step=dyn_cast<BinaryOperator>(iv->getIncomingValueForBlock(bb));
  if(!start || !step || step->getOpcode()!=Instruction::Add || step->getOperand(0)!=iv)return {};
  auto *s=dyn_cast<ConstantInt>(step->getOperand(1));
  if(!s || !s->getZExtValue() || bound->getZExtValue()<=start->getZExtValue())return {};
  uint64_t diff=bound->getZExtValue()-start->getZExtValue(),inc=s->getZExtValue();
  if(diff%inc || bound->getZExtValue()+inc>0x7fffffffu)return {};
  SmallPtrSet<Value *,32> seen;
  std::function<bool(Value *)> independent=[&](Value *v) {
    auto *i=dyn_cast<Instruction>(v);if(!i || i->getParent()!=bb || !seen.insert(v).second)return true;
    if(isa<PHINode>(i))return i==iv;
    for(Value *op:i->operands())if(!independent(op))return false;
    return true;
  };
  for(auto &p:c.panels)for(auto *copy:p.copies)for(Value *v:copy->args())if(!independent(v))return {};
  return Trip{iv,step,start,cmp,br,pre,br->getSuccessor(1),diff/inc+1};
}
inline Candidate peelFinal(Candidate &c,Trip &t) {
  auto *bb=c.bb;auto *f=bb->getParent();DenseMap<Value *,Value *> map;
  auto *epi=BasicBlock::Create(bb->getContext(),"panel.final",f);map[bb]=epi;
  // This eligibility has one block and no debug-value intrinsics. A plain
  // operand map suffices and avoids introducing RTTI-only CallbackVH objects
  // into the HCU LLVM build, whose LLVM libraries were built without RTTI.
  for(auto &i:*bb) {
    auto *n=i.clone();n->insertInto(epi,epi->end());map[&i]=n;
  }
  for(auto &i:*epi)for(unsigned n=0;n<i.getNumOperands();++n)
    if(auto found=map.find(i.getOperand(n));found!=map.end())i.setOperand(n,found->second);
  SmallVector<Instruction *> old;
  for(auto &i:*bb)old.push_back(&i);
  for(auto &p:bb->phis()) {
    auto *clone=cast<PHINode>(map[&p]);auto *back=p.getIncomingValueForBlock(bb);
    clone->replaceAllUsesWith(back);map[&p]=back;clone->eraseFromParent();
  }
  epi->getTerminator()->eraseFromParent();BranchInst::Create(t.exit,epi);
  auto *newBound=ConstantInt::get(t.cmp->getOperand(1)->getType(),cast<ConstantInt>(t.cmp->getOperand(1))->getZExtValue()-cast<ConstantInt>(t.step->getOperand(1))->getZExtValue());
  IRBuilder<> latch(t.br);t.br->setCondition(latch.CreateICmpULT(t.iv,newBound));t.br->setSuccessor(1,epi);
  for(auto &p:t.exit->phis()) {
    int n=p.getBasicBlockIndex(bb);if(n>=0)p.setIncomingBlock(n,epi);
  }
  for(auto *i:old)if(!i->getType()->isVoidTy()) {
    SmallVector<Use *> uses;
    for(auto &u:i->uses())if(auto *user=dyn_cast<Instruction>(u.getUser());user && user->getParent()!=bb && user->getParent()!=epi)uses.push_back(&u);
    for(auto *u:uses)u->set(map[i]);
  }
  Candidate tail;tail.bb=epi;
  for(auto &p:c.panels) {
    Panel q;q.compactStride=p.compactStride;
    for(auto *x:p.copies)q.copies.push_back(cast<CallInst>(map[x]));
    for(auto *x:p.reads)q.reads.push_back(cast<CallInst>(map[x]));
    for(auto *x:p.mmas)q.mmas.push_back(cast<CallInst>(map[x]));
    tail.panels.push_back(std::move(q));
  }
  for(auto *x:c.sync)tail.sync.push_back(cast<CallInst>(map[x]));
  return tail;
}
inline void unrollTwo(BranchInst *br) {
  auto &ctx=br->getContext();
  auto *count=MDNode::get(ctx,{MDString::get(ctx,"llvm.loop.unroll.count"),ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx),2))});
  auto *id=MDNode::getDistinct(ctx,{nullptr,count});id->replaceOperandWith(0,id);br->setMetadata(LLVMContext::MD_loop,id);
}
inline Report run(Module &m, StringRef arch, unsigned waves, unsigned shared,
                  bool stream=false, bool identity=false, bool cross=false,
                  bool compactK=false, bool unroll=false, bool nativeWar=false, unsigned warAfter=0) {
  Report r; r.shared=shared;
  if(warAfter && ((warAfter!=8 && warAfter!=16 && warAfter!=24) || !nativeWar || stream)){r.reason="deferred WAR requires8/16/24MMAC/native WAR/no V stream";return r;}
  if(arch!="gfx938" || waves!=4 || shared!=8192 || m.getDataLayout().getPointerSizeInBits(3)!=32) {
    r.reason="requires gfx938/four waves/exclusive 8KiB panel";return r;
  }
  SmallVector<Candidate,1> candidates;
  for(auto &f:m) if(f.getCallingConv()==CallingConv::AMDGPU_KERNEL) {
    // Every explicit LDS consumer must use the same verified matrix family.
    for(auto &bb:f)for(auto &i:bb) {
      if(auto *l=dyn_cast<LoadInst>(&i);l && l->getPointerAddressSpace()==3){r.reason="other LDS read";return r;}
      if(auto *s=dyn_cast<StoreInst>(&i);s && s->getPointerAddressSpace()==3){r.reason="other LDS write";return r;}
    }
    DominatorTree dt(f); LoopInfo li(dt);
    size_t prior=candidates.size();
    for(auto *loop:li) {
      if(loop->getNumBlocks()!=1 || !loop->getSubLoops().empty())continue;
      auto *bb=loop->getHeader();
      auto *br=dyn_cast<BranchInst>(bb->getTerminator());
      if(!br || !br->isConditional() || (br->getSuccessor(0)!=bb && br->getSuccessor(1)!=bb))continue;
      auto c=analyze(*bb,r.reason);if(c)candidates.push_back(std::move(*c));
    }
    // A single full/tail tile is often completely unrolled by O3. Isolate
    // only its original synchronization-to-last-MMA interval. Ordinary memory
    // inside that interval still rejects; Q input and output accesses outside
    // it are untouched, and the existing whole-kernel exclusive-LDS check holds.
    if(candidates.size()==prior)for(auto &bb:f)if(!li.getLoopFor(&bb)) {
      auto c=analyze(bb,r.reason,true);if(c)candidates.push_back(std::move(*c));
    }
  }
  if(warAfter)for(auto &c:candidates)if(!deferEligible(c,warAfter)){r.reason="DS/MMAC ordering does not prove deferred WAR";return r;}
  for(auto &c:candidates){
    if(warAfter)r.deferred+=c.panels.size();
    if(compactK)r.compacted+=compact(c);
    auto trip=cross?proveTrip(c):std::nullopt;
    if(trip) {
      auto tail=peelFinal(c,*trip);
      std::optional<Candidate> penultimate;
      uint64_t remaining=trip->count-1;
      // Keep the steady count even without asking LLVM for a runtime remainder.
      // The second peeled body is ordered before the already peeled final tile.
      if(unroll && trip->count>=4 && trip->count%2==0) {
        if(auto shorter=proveTrip(c)) {
          penultimate=peelFinal(c,*shorter);--remaining;
        }
      }
      IRBuilder<> pro(trip->pre->getTerminator());
      assembly(pro,"s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier");
      issue(pro,c,0,{},trip->iv,trip->start);issue(pro,c,1,{},trip->iv,trip->start);
      if(stream){r.streamed+=streamV(c);r.streamed+=streamV(tail);}
      transform(c,true,trip->iv,trip->step,nativeWar,warAfter);
      // In the penultimate body the old loop backedge is its current index.
      // Remap that SSA value to +step when prefetching the real final tile.
      if(penultimate)transform(*penultimate,true,trip->step,trip->step,nativeWar,warAfter);
      transform(tail,true,nullptr,nullptr,nativeWar,warAfter);
      if(identity){r.rescaled+=guardRescale(c);if(penultimate)r.rescaled+=guardRescale(*penultimate);r.rescaled+=guardRescale(tail);}
      if(unroll && remaining%2==0){unrollTwo(trip->br);++r.unrolled;}
      ++r.crossed;
    } else {
      if(stream)r.streamed+=streamV(c);transform(c,false,nullptr,nullptr,nativeWar,warAfter);if(identity)r.rescaled+=guardRescale(c);
    }
    ++r.loops;r.panels+=c.panels.size();
  }
  if(r.loops){r.shared=16384;r.reason=r.crossed?"applied; bounded cross-iteration two-slot pipeline with final drain":"applied; iteration-local two-slot pipeline";}
  else if(r.reason.empty())r.reason="no eligible single-block panel loop";
  std::string errors;raw_string_ostream os(errors);
  if(verifyModule(m,&os))throw std::runtime_error("HCU panel pipeline produced invalid IR: "+errors);
  return r;
}
} // namespace hcu_tle_panel
