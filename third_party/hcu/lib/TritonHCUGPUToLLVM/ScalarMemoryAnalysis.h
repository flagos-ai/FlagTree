#ifndef TRITON_HCU_SCALAR_MEMORY_ANALYSIS_H
#define TRITON_HCU_SCALAR_MEMORY_ANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "WarpSpecializeUtils.h"
#include <optional>

namespace mlir::triton::HCU::memory_analysis {
using namespace mlir;
using ::mlir::triton::TritonLLVMOpBuilder;
static std::optional<int64_t> constant(Value value) {
  value = resolveWarpSpecializeCapture(value);
  auto c = value.getDefiningOp<arith::ConstantOp>();
  auto a = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr{};
  if (!a || a.getValue().getBitWidth() > 64) return std::nullopt;
  return a.getValue().getSExtValue();
}
// Scalar SSA is not automatically CTA-uniform: CF PHIs also require uniform
// control. Reject every unsupported leaf, including thread/lane IDs and loads.
static bool uniformValue(Value v,llvm::SmallPtrSetImpl<void *> &seen) {
  if(!v.getType().isIntOrIndex())return false;
  Value capture=resolveWarpSpecializeCapture(v);
  if(capture!=v)return uniformValue(capture,seen);
  if(!seen.insert(v.getAsOpaquePointer()).second)return true;
  if(auto arg=dyn_cast<BlockArgument>(v)) {
    Block *dest=arg.getOwner();unsigned n=arg.getArgNumber();
    if(auto f=dyn_cast<triton::FuncOp>(dest->getParentOp()))
      if(dest==&f.getBody().front())return true; // scalar kernel ABI
    if(dest->hasNoPredecessors())return false;
    for(Block *pred:dest->getPredecessors()) {
      Operation *t=pred->getTerminator();bool edge=false;
      if(auto br=dyn_cast<cf::BranchOp>(t)) {
        if(br.getDest()!=dest || !uniformValue(br.getDestOperands()[n],seen))return false;edge=true;
      } else if(auto br=dyn_cast<cf::CondBranchOp>(t)) {
        if(!uniformValue(br.getCondition(),seen))return false;
        if(br.getTrueDest()==dest) {if(!uniformValue(br.getTrueDestOperands()[n],seen))return false;edge=true;}
        if(br.getFalseDest()==dest) {if(!uniformValue(br.getFalseDestOperands()[n],seen))return false;edge=true;}
      }
      if(!edge)return false;
    }
    return true;
  }
  Operation *op=v.getDefiningOp();if(!op)return false;
  if(isa<triton::GetProgramIdOp>(op))return true;
  if(!op->getName().getStringRef().starts_with("arith.") || op->getNumRegions())return false;
  for(Value operand:op->getOperands())if(!uniformValue(operand,seen))return false;
  return true;
}
static bool uniform(Value v,Operation *anchor=nullptr) {
  llvm::SmallPtrSet<void *,32> seen;
  if(!uniformValue(v,seen))return false;
  auto f=v.getParentBlock()->getParentOp()->getParentOfType<triton::FuncOp>();
  if(!f)f=dyn_cast<triton::FuncOp>(v.getParentBlock()->getParentOp());
  if(!f)return false;
  bool result=true;f.walk([&](cf::CondBranchOp br) {
    // Control in another partition does not determine this instruction's
    // active lanes. Retain all controlling/ancestor-region CF conditions.
    if(anchor && br->getParentRegion()!=anchor->getParentRegion() &&
       !br->getParentRegion()->isAncestor(anchor->getParentRegion()))
      return;
    llvm::SmallPtrSet<void *,32> control;
    result&=uniformValue(br.getCondition(),control);
  });
  return result;
}
// Power-of-two divisibility is invariant under integer wrap, so it does
// not require nowrap assumptions. CF cycles must retain a divisible entry seed.
static bool multipleOf(Value v,unsigned n,llvm::SmallPtrSetImpl<void *> &active) {
  if(!v.getType().isInteger() || !n || (n&(n-1)))return false;
  Value capture=resolveWarpSpecializeCapture(v);
  if(capture!=v)return multipleOf(capture,n,active);
  if(auto c=constant(v))return uint64_t(*c)%n==0;
  if(!active.insert(v.getAsOpaquePointer()).second)return true;
  bool answer=[&]() {
    if(auto arg=dyn_cast<BlockArgument>(v)) {
      Block *dest=arg.getOwner();unsigned k=arg.getArgNumber();if(dest->hasNoPredecessors())return false;
      for(Block *pred:dest->getPredecessors()) {
        Operation *t=pred->getTerminator();bool edge=false;
        if(auto br=dyn_cast<cf::BranchOp>(t)) {
          if(br.getDest()!=dest || !multipleOf(br.getDestOperands()[k],n,active))return false;edge=true;
        } else if(auto br=dyn_cast<cf::CondBranchOp>(t)) {
          if(br.getTrueDest()==dest) {if(!multipleOf(br.getTrueDestOperands()[k],n,active))return false;edge=true;}
          if(br.getFalseDest()==dest) {if(!multipleOf(br.getFalseDestOperands()[k],n,active))return false;edge=true;}
        }
        if(!edge)return false;
      }
      return true;
    }
    Operation *op=v.getDefiningOp();if(!op)return false;
    auto multiple=[&](Value x){return multipleOf(x,n,active);};
    if(isa<arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp>(op)) {
      auto src=op->getOperand(0);auto srcTy=dyn_cast<IntegerType>(src.getType());auto dstTy=dyn_cast<IntegerType>(v.getType());
      return srcTy && dstTy && uint64_t(n)<=(uint64_t(1)<<std::min<unsigned>(31,std::min(srcTy.getWidth(),dstTy.getWidth()))) && multiple(src);
    }
    if(isa<arith::AddIOp,arith::SubIOp,arith::OrIOp,arith::XOrIOp,arith::MaxSIOp,arith::MinSIOp,arith::MaxUIOp,arith::MinUIOp>(op))
      return multiple(op->getOperand(0)) && multiple(op->getOperand(1));
    if(auto mul=dyn_cast<arith::MulIOp>(op))return multiple(mul.getLhs()) || multiple(mul.getRhs());
    if(isa<arith::RemSIOp,arith::RemUIOp>(op)) {
      auto c=constant(op->getOperand(1));return c && *c>0 && uint64_t(*c)%n==0 && multiple(op->getOperand(0));
    }
    if(auto select=dyn_cast<arith::SelectOp>(op))return multiple(select.getTrueValue()) && multiple(select.getFalseValue());
    if(auto bit=dyn_cast<arith::AndIOp>(op)) {
      auto c=constant(bit.getRhs());return (c && (uint64_t(*c)&(n-1))==0) || multiple(bit.getLhs()) || multiple(bit.getRhs());
    }
    return false;
  }();
  active.erase(v.getAsOpaquePointer());return answer;
}
static bool multipleOf(Value v,unsigned n) {llvm::SmallPtrSet<void *,32> active;return multipleOf(v,n,active);}
static Value uniformSaturatingSub(Location loc,ConversionPatternRewriter &rw,Value lhs,Value rhs) {
  return LLVM::InlineAsmOp::create(rw,loc,lhs.getType(),ValueRange{lhs,rhs},
      "s_sub_u32 $0, $1, $2\n\ts_cmp_ge_u32 $1, $2\n\ts_cselect_b32 $0, $0, 0",
      "=&s,s,s,~{scc}",false,false,LLVM::TailCallKind::None,
      LLVM::AsmDialectAttr::get(rw.getContext(),LLVM::AsmDialect::AD_ATT),ArrayAttr::get(rw.getContext(),{})).getRes();
}
// Exact signed intervals from ordinary scalar SSA and a canonical counted CF
// loop. No launch bounds, imported certificates, assumptions, shapes or tuning
// configurations are used. Arithmetic that may wrap loses its range proof.
struct IndexBounds { int64_t lo, hi; };
static bool boundsFit(__int128 lo,__int128 hi,Type type) {
  auto integer=dyn_cast<IntegerType>(type);
  if(!integer || integer.getWidth()<2 || integer.getWidth()>64)return false;
  __int128 limit=__int128(1)<<(integer.getWidth()-1);
  return lo>=-limit && hi<limit && lo<=hi;
}
static std::optional<int64_t> counterStep(Value next,Value phi) {
  auto add=next.getDefiningOp<arith::AddIOp>();if(!add)return std::nullopt;
  if(add.getLhs()==phi)return constant(add.getRhs());
  if(add.getRhs()==phi)return constant(add.getLhs());
  return std::nullopt;
}
static std::optional<IndexBounds> countedPhiBounds(
    BlockArgument arg,Operation *anchor) {
  Block *header=arg.getOwner();
  auto test=dyn_cast<cf::CondBranchOp>(header->getTerminator());
  if(!test || test.getTrueDest()!=anchor->getBlock() ||
      !test.getTrueDestOperands().empty())return std::nullopt;
  Block *body=test.getTrueDest();
  if(body==header || body->getSinglePredecessor()!=header ||
      test.getFalseDest()==body || test.getFalseDest()==header)return std::nullopt;
  auto cmp=test.getCondition().getDefiningOp<arith::CmpIOp>();
  auto iv=cmp?dyn_cast<BlockArgument>(cmp.getLhs()):BlockArgument{};
  if(!iv || iv.getOwner()!=header || cmp->getBlock()!=header ||
      cmp.getPredicate()!=arith::CmpIPredicate::slt)return std::nullopt;
  auto bound=constant(cmp.getRhs());if(!bound)return std::nullopt;
  auto back=dyn_cast<cf::BranchOp>(body->getTerminator());
  if(!back || back.getDest()!=header ||
      back.getDestOperands().size()!=header->getNumArguments())return std::nullopt;
  auto predecessors=llvm::to_vector(header->getPredecessors());
  if(predecessors.size()!=2)return std::nullopt;
  Block *entry=nullptr;
  for(Block *pred:predecessors)if(pred!=body) {
    if(entry)return std::nullopt;
    entry=pred;
  }
  if(!entry || entry==header)return std::nullopt;
  auto enter=dyn_cast<cf::BranchOp>(entry->getTerminator());
  if(!enter || enter.getDest()!=header ||
      enter.getDestOperands().size()!=header->getNumArguments())return std::nullopt;
  unsigned primary=iv.getArgNumber(),secondary=arg.getArgNumber();
  auto initial=constant(enter.getDestOperands()[primary]);
  auto step=counterStep(back.getDestOperands()[primary],iv);
  auto seed=constant(enter.getDestOperands()[secondary]);
  auto advance=counterStep(back.getDestOperands()[secondary],arg);
  if(!initial || !step || *step<=0 || !seed || !advance || *initial>=*bound)
    return std::nullopt;
  __int128 trips=(__int128(*bound)-*initial+*step-1)/ *step;
  __int128 finalPrimary=__int128(*initial)+trips* *step;
  __int128 finalSecondary=__int128(*seed)+trips* *advance;
  if(!boundsFit(*initial,finalPrimary,iv.getType()) ||
      !boundsFit(std::min<__int128>(*seed,finalSecondary),
                       std::max<__int128>(*seed,finalSecondary),arg.getType()))
    return std::nullopt;
  __int128 last=__int128(*seed)+(trips-1)* *advance;
  return IndexBounds{int64_t(std::min<__int128>(*seed,last)),
                           int64_t(std::max<__int128>(*seed,last))};
}
static std::optional<IndexBounds> indexBounds(
    Value value,Operation *anchor,llvm::SmallPtrSetImpl<void *> &active,unsigned &fuel) {
  if(!fuel)return std::nullopt;--fuel;
  Value capture=resolveWarpSpecializeCapture(value);
  if(capture!=value)return indexBounds(capture,anchor,active,fuel);
  auto integer=dyn_cast<IntegerType>(value.getType());
  if(!integer || integer.getWidth()<2 || integer.getWidth()>64)return std::nullopt;
  if(auto c=constant(value))return IndexBounds{*c,*c};
  if(!active.insert(value.getAsOpaquePointer()).second)return std::nullopt;
  auto result=[&]()->std::optional<IndexBounds> {
    if(auto arg=dyn_cast<BlockArgument>(value))return countedPhiBounds(arg,anchor);
    Operation *op=value.getDefiningOp();if(!op)return std::nullopt;
    auto range=[&](Value v){return indexBounds(v,anchor,active,fuel);};
    if(isa<arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp>(op)) {
      auto source=range(op->getOperand(0));
      if(!source || (isa<arith::ExtUIOp>(op) && source->lo<0) ||
          !boundsFit(source->lo,source->hi,value.getType()))return std::nullopt;
      return source;
    }
    if(auto select=dyn_cast<arith::SelectOp>(op)) {
      auto a=range(select.getTrueValue()),b=range(select.getFalseValue());
      if(!a || !b)return std::nullopt;
      return IndexBounds{std::min(a->lo,b->lo),std::max(a->hi,b->hi)};
    }
    if(op->getNumOperands()!=2)return std::nullopt;
    if(isa<arith::RemUIOp,arith::DivUIOp>(op)) {
      auto c=constant(op->getOperand(1));if(!c)return std::nullopt;
      unsigned width=integer.getWidth();
      __int128 divisor=uint64_t(*c) & (~uint64_t(0) >> (64-width));
      if(!divisor)return std::nullopt;
      // Unsigned arithmetic consumes bit patterns, including negative signed
      // inputs. Unknown/cross-sign inputs cover the full unsigned domain.
      __int128 modulus=__int128(1)<<width,lo=0,hi=modulus-1;
      if(auto source=range(op->getOperand(0))) {
        if(source->lo>=0){lo=source->lo;hi=source->hi;}
        else if(source->hi<0){lo=modulus+source->lo;hi=modulus+source->hi;}
      }
      if(isa<arith::DivUIOp>(op)){lo/=divisor;hi/=divisor;}
      else {lo=0;hi=std::min(hi,divisor-1);}
      // The consumer's interval is signed: a wrapped/nonrepresentable result
      // stays unknown. No launch bounds or nowrap assumption is introduced.
      if(!boundsFit(lo,hi,value.getType()))return std::nullopt;
      return IndexBounds{int64_t(lo),int64_t(hi)};
    }
    auto a=range(op->getOperand(0)),b=range(op->getOperand(1));
    if(isa<arith::MaxSIOp,arith::MinSIOp>(op)) {
      // A clamp bounds an otherwise unknown integer too. Use its full signed
      // source domain, never an assumed nonnegative program/loop index.
      auto full=[&](Value v) {
        unsigned w=cast<IntegerType>(v.getType()).getWidth();
        __int128 limit=__int128(1)<<(w-1);
        return IndexBounds{int64_t(-limit),int64_t(limit-1)};
      };
      if(!a)a=full(op->getOperand(0));if(!b)b=full(op->getOperand(1));
      if(isa<arith::MaxSIOp>(op))return IndexBounds{std::max(a->lo,b->lo),std::max(a->hi,b->hi)};
      return IndexBounds{std::min(a->lo,b->lo),std::min(a->hi,b->hi)};
    }
    if(!a || !b)return std::nullopt;
    __int128 lo,hi;
    if(isa<arith::AddIOp>(op)){lo=__int128(a->lo)+b->lo;hi=__int128(a->hi)+b->hi;}
    else if(isa<arith::SubIOp>(op)){lo=__int128(a->lo)-b->hi;hi=__int128(a->hi)-b->lo;}
    else if(isa<arith::MulIOp>(op)) {
      __int128 v[]={__int128(a->lo)*b->lo,__int128(a->lo)*b->hi,
                    __int128(a->hi)*b->lo,__int128(a->hi)*b->hi};
      lo=*std::min_element(v,v+4);hi=*std::max_element(v,v+4);
    } else if(isa<arith::DivSIOp,arith::RemSIOp>(op) && a->lo>=0 &&
              b->lo==b->hi && b->lo>0) {
      if(isa<arith::DivSIOp>(op)){lo=a->lo/b->lo;hi=a->hi/b->lo;}
      else {lo=0;hi=std::min(a->hi,b->lo-1);}
    } else return std::nullopt;
    if(!boundsFit(lo,hi,value.getType()))return std::nullopt;
    return IndexBounds{int64_t(lo),int64_t(hi)};
  }();
  active.erase(value.getAsOpaquePointer());return result;
}
static std::optional<IndexBounds> indexBounds(Value v,Operation *anchor) {
  llvm::SmallPtrSet<void *,32> active;unsigned fuel=512;return indexBounds(v,anchor,active,fuel);
}


} // namespace mlir::triton::HCU::memory_analysis
#endif
