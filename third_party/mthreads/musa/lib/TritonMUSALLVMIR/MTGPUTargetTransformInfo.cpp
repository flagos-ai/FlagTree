#include "TritonMUSALLVMIR/MTGPUTargetTransformInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include <algorithm>
#include <memory>
#include <optional>

using namespace llvm;

namespace {

Type *getScalarElementType(Type *Ty) {
  if (auto *VecTy = dyn_cast<VectorType>(Ty))
    return VecTy->getElementType();
  return Ty;
}

bool isPackedFOPType(Type *Ty) {
  Ty = getScalarElementType(Ty);
  return Ty->isHalfTy() || Ty->isFloatTy();
}

bool isPH1Target(StringRef TargetTriple, StringRef Processor) {
  return TargetTriple == "mtgpu-mt-musa" && Processor == "mp_31";
}

unsigned getFixedWidthNumElements(Type *Ty) {
  if (auto *VecTy = dyn_cast<FixedVectorType>(Ty))
    return VecTy->getNumElements();
  return 1;
}

InstructionCost getBundledIssueCost(unsigned NumElts) {
  if (NumElts <= 1)
    return 1;
  return NumElts - NumElts / 4;
}

InstructionCost getSingleSourcePermuteCost(unsigned NumElts) {
  if (NumElts <= 1)
    return 0;
  return std::max(1U, NumElts / 2);
}

InstructionCost getTwoSourcePermuteCost(unsigned NumElts) {
  if (NumElts <= 1)
    return 0;
  return NumElts;
}

InstructionCost getSubvectorShuffleCost(unsigned NumElts,
                                        const VectorType *SubTy) {
  if (auto *SubVT = dyn_cast_or_null<FixedVectorType>(SubTy))
    NumElts = std::max(NumElts, SubVT->getNumElements());
  return std::max(1U, NumElts / 2);
}

class MTGPUTTIImpl final
    : public TargetTransformInfoImplCRTPBase<MTGPUTTIImpl> {
  using BaseT = TargetTransformInfoImplCRTPBase<MTGPUTTIImpl>;
  using TTI = TargetTransformInfo;

public:
  explicit MTGPUTTIImpl(const Function &F) : BaseT(F.getDataLayout()) {}

  using BaseT::getArithmeticInstrCost;
  using BaseT::getCmpSelInstrCost;
  using BaseT::getScalarizationOverhead;
  using BaseT::getShuffleCost;
  using BaseT::getVectorInstrCost;

  bool hasBranchDivergence(const Function *F = nullptr) const override {
    return false;
  }

  unsigned getNumberOfRegisters(unsigned ClassID) const override { return 1; }
  TypeSize getRegisterBitWidth(TTI::RegisterKind K) const override {
    return TypeSize::getFixed(32);
  }
  unsigned getMinVectorRegisterBitWidth() const override { return 32; }

  unsigned getMaximumVF(unsigned ElemWidth, unsigned Opcode) const override {
    switch (Opcode) {
    case Instruction::FNeg:
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FCmp:
    case Instruction::Select:
    case Instruction::InsertElement:
      return (ElemWidth == 16 || ElemWidth == 32) ? 8U : 1U;
    default:
      return 1;
    }
  }

  InstructionCost getArithmeticInstrCost(
      unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
      TTI::OperandValueInfo Opd1Info, TTI::OperandValueInfo Opd2Info,
      ArrayRef<const Value *> Args,
      const Instruction *CxtI = nullptr) const override {
    Type *ScalarTy = Ty->getScalarType();
    unsigned NumElts = getFixedWidthNumElements(Ty);

    switch (Opcode) {
    default:
      return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Opd1Info,
                                           Opd2Info, Args, CxtI);
    case Instruction::Add:
    case Instruction::Mul:
    case Instruction::Xor:
    case Instruction::Or:
    case Instruction::And:
      if (ScalarTy->isIntegerTy(64))
        return 2;
      return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Opd1Info,
                                           Opd2Info, Args, CxtI);
    case Instruction::FMul:
      if (CxtI && CxtI->hasOneUse()) {
        if (const auto *FAdd = dyn_cast<BinaryOperator>(*CxtI->user_begin())) {
          unsigned UserOpcode = FAdd->getOpcode();
          if ((UserOpcode == Instruction::FAdd ||
               UserOpcode == Instruction::FSub) &&
              (ScalarTy->isFloatTy() || ScalarTy->isHalfTy()))
            return TTI::TCC_Free;
        }
      }
      [[fallthrough]];
    case Instruction::FAdd:
    case Instruction::FSub:
      if ((ScalarTy->isFloatTy() || ScalarTy->isHalfTy()) && NumElts > 1)
        return getBundledIssueCost(NumElts);
      return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Opd1Info,
                                           Opd2Info, Args, CxtI);
    }
  }

  InstructionCost
  getShuffleCost(TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
                 ArrayRef<int> Mask, TTI::TargetCostKind CostKind, int Index,
                 VectorType *SubTy, ArrayRef<const Value *> Args = {},
                 const Instruction *CxtI = nullptr) const override {
    VectorType *CostTy = Kind == TTI::SK_InsertSubvector ? DstTy : SrcTy;
    if (!CostTy || !isa<FixedVectorType>(CostTy) || !isPackedFOPType(CostTy))
      return BaseT::getShuffleCost(Kind, DstTy, SrcTy, Mask, CostKind, Index,
                                   SubTy, Args, CxtI);
    unsigned NumElts = cast<FixedVectorType>(CostTy)->getNumElements();
    switch (Kind) {
    case TTI::SK_Broadcast:
      return 1;
    case TTI::SK_PermuteSingleSrc:
      return getSingleSourcePermuteCost(NumElts);
    case TTI::SK_PermuteTwoSrc:
      return getTwoSourcePermuteCost(NumElts);
    case TTI::SK_ExtractSubvector:
    case TTI::SK_InsertSubvector:
      return getSubvectorShuffleCost(NumElts, SubTy);
    default:
      return BaseT::getShuffleCost(Kind, DstTy, SrcTy, Mask, CostKind, Index,
                                   SubTy, Args, CxtI);
    }
  }

  InstructionCost
  getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy,
                     CmpInst::Predicate VecPred, TTI::TargetCostKind CostKind,
                     TTI::OperandValueInfo Op1Info,
                     TTI::OperandValueInfo Op2Info,
                     const Instruction *I = nullptr) const override {
    if (!isa<FixedVectorType>(ValTy) || !isPackedFOPType(ValTy))
      return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                       Op1Info, Op2Info, I);

    switch (Opcode) {
    case Instruction::FCmp:
    case Instruction::Select:
      return getBundledIssueCost(
          cast<FixedVectorType>(ValTy)->getNumElements());
    default:
      return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                       Op1Info, Op2Info, I);
    }
  }

  InstructionCost getScalarizationOverhead(
      VectorType *Ty, const APInt &DemandedElts, bool Insert, bool Extract,
      TTI::TargetCostKind CostKind, bool ForPoisonSrc = true,
      ArrayRef<Value *> VL = {}) const override {
    if (!isa<FixedVectorType>(Ty) || !isPackedFOPType(Ty))
      return BaseT::getScalarizationOverhead(Ty, DemandedElts, Insert, Extract,
                                             CostKind, ForPoisonSrc, VL);

    unsigned NumLanes = cast<FixedVectorType>(Ty)->getNumElements();
    unsigned DemandedLanes = DemandedElts.popcount();
    if (DemandedLanes == 0)
      return 0;
    DemandedLanes = std::min(NumLanes, DemandedLanes);

    InstructionCost Cost = 0;
    if (Insert) {
      if (ForPoisonSrc) {
        Cost += DemandedLanes;
      } else {
        unsigned NumDemandedBits =
            std::min<unsigned>(DemandedElts.getBitWidth(), NumLanes);
        for (unsigned I = 0; I != NumDemandedBits; ++I) {
          if (!DemandedElts[I])
            continue;
          const Value *InsertedVal = I < VL.size() ? VL[I] : nullptr;
          Cost +=
              getVectorInstrCost(Instruction::InsertElement, Ty, CostKind, I,
                                 Constant::getNullValue(Ty), InsertedVal);
        }
      }
    }
    if (Extract)
      Cost += DemandedLanes;
    return Cost;
  }

  std::optional<InstructionCost>
  getPackedElementAccessCost(unsigned Opcode, Type *ValTy,
                             unsigned Index) const {
    if (Opcode != Instruction::ExtractElement &&
        Opcode != Instruction::InsertElement)
      return std::nullopt;
    if (!isa<FixedVectorType>(ValTy) || !isPackedFOPType(ValTy))
      return std::nullopt;
    return Index == ~0u ? 2 : 0;
  }

  InstructionCost getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                     TTI::TargetCostKind CostKind,
                                     unsigned Index, const Value *Op0,
                                     const Value *Op1) const override {
    if (auto Cost = getPackedElementAccessCost(Opcode, ValTy, Index))
      return *Cost;
    return BaseT::getVectorInstrCost(Opcode, ValTy, CostKind, Index, Op0, Op1);
  }

  InstructionCost getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                     TTI::TargetCostKind CostKind,
                                     unsigned Index, Value *Scalar,
                                     ArrayRef<std::tuple<Value *, User *, int>>
                                         ScalarUserAndIdx) const override {
    if (auto Cost = getPackedElementAccessCost(Opcode, ValTy, Index))
      return *Cost;
    return BaseT::getVectorInstrCost(Opcode, ValTy, CostKind, Index, Scalar,
                                     ScalarUserAndIdx);
  }

  InstructionCost getVectorInstrCost(const Instruction &I, Type *ValTy,
                                     TTI::TargetCostKind CostKind,
                                     unsigned Index) const override {
    if (auto Cost = getPackedElementAccessCost(I.getOpcode(), ValTy, Index))
      return *Cost;
    return BaseT::getVectorInstrCost(I, ValTy, CostKind, Index);
  }
};

} // namespace

bool llvm::mtgpu::registerMTGPUTargetIRAnalysis(FunctionAnalysisManager &FAM,
                                                StringRef TargetTriple,
                                                StringRef Processor) {
  if (!isPH1Target(TargetTriple, Processor))
    return false;

  return FAM.registerPass([] {
    return TargetIRAnalysis([](const Function &F) {
      return TargetTransformInfo(std::make_unique<MTGPUTTIImpl>(F));
    });
  });
}
