#ifndef TRITON_HCU_SHARED_ADDRESS_EVALUATOR_H
#define TRITON_HCU_SHARED_ADDRESS_EVALUATOR_H

#include "llvm/ADT/MapVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include <cstdint>
#include <optional>

namespace flagtree::hcu::memory_schedule {
using namespace llvm;

// Only bit-accurate integer/AS3 address conversions are modeled. In
// particular FP casts and addrspacecast must never become integer identity.
static bool addressCast(unsigned opcode, Type *from, Type *to) {
  auto as3=[](Type *t) {auto *p=dyn_cast<PointerType>(t);return p && p->getAddressSpace()==3;};
  switch(opcode) {
  case Instruction::Trunc: case Instruction::ZExt: case Instruction::SExt:
    return from->isIntegerTy() && to->isIntegerTy();
  case Instruction::PtrToInt:return as3(from) && to->isIntegerTy();
  case Instruction::IntToPtr:return from->isIntegerTy() && as3(to);
  case Instruction::BitCast:return as3(from) && as3(to);
  default:return false;
  }
}
// Exact finite evaluation of the address-only slice. No loaded value, loop IV,
// resource descriptor, unknown call or user certificate supplies an address.
struct SharedAddressEvaluator {
  static bool named(Instruction *inst,StringRef prefix) {
    auto *call=dyn_cast<CallBase>(inst);
    auto *fn=call?call->getCalledFunction():nullptr;
    return fn && fn->getName().starts_with(prefix);
  }
  static InlineAsm *assembly(Instruction *inst) {
    auto *call=dyn_cast<CallBase>(inst);
    return call?dyn_cast<InlineAsm>(call->getCalledOperand()):nullptr;
  }
  const DataLayout &dl; GlobalVariable *shared; unsigned threads;
  std::optional<uint64_t> operator()(Value *v,unsigned tid,unsigned depth=0) {
    if(depth>160)return std::nullopt;
    if(auto *c=dyn_cast<ConstantInt>(v))return c->getValue().zextOrTrunc(64).getZExtValue();
    if(v==shared)return uint64_t(0);
    if(auto *g=dyn_cast<GEPOperator>(v)) {
      auto b=(*this)(g->getPointerOperand(),tid,depth+1);if(!b)return std::nullopt;
#if LLVM_VERSION_MAJOR >= 22
      SmallMapVector<Value *,APInt,4> variables;
#else
      MapVector<Value *,APInt> variables;
#endif
      APInt constant(32,0);
      if(!g->collectOffset(dl,32,variables,constant))return std::nullopt;
      uint64_t result=*b+constant.sextOrTrunc(64).getZExtValue();
      for(auto &x:variables) {auto n=(*this)(x.first,tid,depth+1);if(!n)return std::nullopt;result+=*n*x.second.sextOrTrunc(64).getZExtValue();}
      return result&0xffffffffu;
    }
    if(auto *c=dyn_cast<ConstantExpr>(v)) {
      if(c->isCast()) {
        if(!addressCast(c->getOpcode(),c->getOperand(0)->getType(),c->getType()))return std::nullopt;
        auto x=(*this)(c->getOperand(0),tid,depth+1);if(!x)return std::nullopt;
        unsigned bits=c->getType()->isIntegerTy()?c->getType()->getIntegerBitWidth():32;
        unsigned src=c->getOperand(0)->getType()->isIntegerTy()?c->getOperand(0)->getType()->getIntegerBitWidth():32;
        APInt a(src,*x);return (c->getOpcode()==Instruction::SExt?a.sextOrTrunc(bits):a.zextOrTrunc(bits)).zextOrTrunc(64).getZExtValue();
      }
      return std::nullopt;
    }
    auto *i=dyn_cast<Instruction>(v);if(!i)return std::nullopt;
    if(named(i,"llvm.amdgcn.workitem.id.x"))return tid;
    if(named(i,"llvm.amdgcn.readfirstlane."))return (*this)(cast<CallBase>(i)->getArgOperand(0),(tid/64)*64,depth+1);
    // The existing LDS-DMA alias workaround intentionally hides pointer
    // provenance with one empty tied VGPR identity. It changes no address bit,
    // including packed m0 rotation fields. Do not infer identity for untied,
    // side-effecting, bundled, wider or otherwise opaque assembly.
    if(auto *c=dyn_cast<CallInst>(i))if(auto *a=assembly(i))
      if(a->getAsmString().empty() && a->getConstraintString()=="=v,0" &&
         !a->hasSideEffects() && !c->getNumOperandBundles() &&
         c->arg_size()==1 && c->getType()->isIntegerTy(32) &&
         c->getArgOperand(0)->getType()->isIntegerTy(32))
        return (*this)(c->getArgOperand(0),tid,depth+1);

    unsigned bits=i->getType()->isIntegerTy()?i->getType()->getIntegerBitWidth():32;
    if(bits>64)return std::nullopt;
    auto get=[&](Value *x){return (*this)(x,tid,depth+1);};
    if(auto *c=dyn_cast<CastInst>(i)) {
      if(!addressCast(c->getOpcode(),c->getSrcTy(),c->getDestTy()))return std::nullopt;
      auto x=get(c->getOperand(0));if(!x)return std::nullopt;
      unsigned src=c->getSrcTy()->isIntegerTy()?c->getSrcTy()->getIntegerBitWidth():32;
      APInt a(src,*x);return (c->getOpcode()==Instruction::SExt?a.sextOrTrunc(bits):a.zextOrTrunc(bits)).zextOrTrunc(64).getZExtValue();
    }
    if(auto *cmp=dyn_cast<ICmpInst>(i)) {
      auto *type=dyn_cast<IntegerType>(cmp->getOperand(0)->getType());
      if(!type || type->getBitWidth()>64)return std::nullopt;
      auto x=get(cmp->getOperand(0)),y=get(cmp->getOperand(1));
      if(!x||!y)return std::nullopt;
      APInt lhs(type->getBitWidth(),*x),rhs(type->getBitWidth(),*y);
      switch(cmp->getPredicate()) {
      case CmpInst::ICMP_EQ:return lhs==rhs;
      case CmpInst::ICMP_NE:return lhs!=rhs;
      case CmpInst::ICMP_UGT:return lhs.ugt(rhs);
      case CmpInst::ICMP_UGE:return lhs.uge(rhs);
      case CmpInst::ICMP_ULT:return lhs.ult(rhs);
      case CmpInst::ICMP_ULE:return lhs.ule(rhs);
      case CmpInst::ICMP_SGT:return lhs.sgt(rhs);
      case CmpInst::ICMP_SGE:return lhs.sge(rhs);
      case CmpInst::ICMP_SLT:return lhs.slt(rhs);
      case CmpInst::ICMP_SLE:return lhs.sle(rhs);
      default:return std::nullopt;
      }
    }
    if(auto *select=dyn_cast<SelectInst>(i)) {
      if(!select->getType()->isIntegerTy() ||
         !select->getCondition()->getType()->isIntegerTy(1))return std::nullopt;
      auto condition=get(select->getCondition());
      if(!condition || *condition>1)return std::nullopt;
      return get(*condition?select->getTrueValue():select->getFalseValue());
    }
    if(auto *b=dyn_cast<BinaryOperator>(i)) {
      auto x=get(b->getOperand(0)),y=get(b->getOperand(1));if(!x||!y)return std::nullopt;
      APInt a(bits,*x),c(bits,*y),r(bits,0);switch(b->getOpcode()) {
        case Instruction::Add:r=a+c;break;case Instruction::Sub:r=a-c;break;case Instruction::Mul:r=a*c;break;
        case Instruction::And:r=a&c;break;case Instruction::Or:r=a|c;break;case Instruction::Xor:r=a^c;break;
        case Instruction::Shl:if(*y>=bits)return std::nullopt;r=a.shl(*y);break;
        case Instruction::LShr:if(*y>=bits)return std::nullopt;r=a.lshr(*y);break;
        case Instruction::AShr:if(*y>=bits)return std::nullopt;r=a.ashr(*y);break;
        default:return std::nullopt;
      }return r.zextOrTrunc(64).getZExtValue();
    }
    return std::nullopt;
  }
};

} // namespace flagtree::hcu::memory_schedule
#endif
