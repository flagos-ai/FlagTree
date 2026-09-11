#include "TritonMUSALLVMIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace mlir::triton::musa;

namespace {

void copyIntermediateLocation(Instruction *Dst, const Instruction &Src) {
  Dst->copyMetadata(Src, {LLVMContext::MD_dbg});
}

void copyReplacementMetadata(Instruction *Dst, const Instruction &Src) {
  Dst->copyMetadata(Src, {LLVMContext::MD_dbg, LLVMContext::MD_annotation,
                          LLVMContext::MD_nosanitize, LLVMContext::MD_fpmath});
}

Value *getHalfVectorIncomingForFPExtPhiRepair(
    Value *V, FixedVectorType *HalfVecTy,
    SmallVectorImpl<Instruction *> &DeadCasts,
    SmallPtrSetImpl<Instruction *> &DeadCastSet) {
  if (auto *FPExt = dyn_cast<FPExtInst>(V)) {
    if (FPExt->getSrcTy() == HalfVecTy && FPExt->hasOneUse()) {
      if (DeadCastSet.insert(FPExt).second)
        DeadCasts.push_back(FPExt);
      return FPExt->getOperand(0);
    }
    return nullptr;
  }

  if (auto *C = dyn_cast<Constant>(V)) {
    if (auto *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::FPExt &&
          CE->getOperand(0)->getType() == HalfVecTy)
        return CE->getOperand(0);
      return nullptr;
    }
    if (isa<PoisonValue>(C))
      return PoisonValue::get(HalfVecTy);
    if (isa<UndefValue>(C))
      return UndefValue::get(HalfVecTy);
    if (C->isNullValue())
      return Constant::getNullValue(HalfVecTy);
  }

  return nullptr;
}

bool repairFPExtPhiCanonicalization(Function &F) {
  SmallVector<PHINode *, 8> Worklist;
  for (BasicBlock &BB : F) {
    for (PHINode &Phi : BB.phis())
      Worklist.push_back(&Phi);
  }

  bool Changed = false;
  for (PHINode *Phi : Worklist) {
    auto *FloatVecTy = dyn_cast<FixedVectorType>(Phi->getType());
    if (!FloatVecTy || !FloatVecTy->getElementType()->isFloatTy())
      continue;

    unsigned NumElts = FloatVecTy->getNumElements();
    if (NumElts != 4 && NumElts != 8)
      continue;

    if (any_of(Phi->users(), [](User *U) { return isa<PHINode>(U); }))
      continue;

    auto *HalfVecTy =
        FixedVectorType::get(Type::getHalfTy(F.getContext()), NumElts);
    SmallVector<Value *, 4> HalfIncomings;
    SmallVector<Instruction *, 4> DeadCasts;
    SmallPtrSet<Instruction *, 4> DeadCastSet;
    HalfIncomings.reserve(Phi->getNumIncomingValues());

    bool CanRepair = true;
    for (Value *Incoming : Phi->incoming_values()) {
      Value *HalfIncoming = getHalfVectorIncomingForFPExtPhiRepair(
          Incoming, HalfVecTy, DeadCasts, DeadCastSet);
      if (!HalfIncoming) {
        CanRepair = false;
        break;
      }
      HalfIncomings.push_back(HalfIncoming);
    }
    if (!CanRepair)
      continue;

    BasicBlock::iterator InsertPt = Phi->getParent()->getFirstInsertionPt();
    if (InsertPt == Phi->getParent()->end())
      continue;

    PHINode *HalfPhi =
        PHINode::Create(HalfVecTy, Phi->getNumIncomingValues(),
                        Phi->getName() + ".half", Phi->getIterator());
    copyIntermediateLocation(HalfPhi, *Phi);
    for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I)
      HalfPhi->addIncoming(HalfIncomings[I], Phi->getIncomingBlock(I));

    IRBuilder<> Builder(Phi->getParent(), InsertPt);
    Builder.SetCurrentDebugLocation(Phi->getDebugLoc());
    Value *Ext = Builder.CreateFPExt(HalfPhi, FloatVecTy, Phi->getName());
    if (auto *ExtInst = dyn_cast<Instruction>(Ext))
      copyReplacementMetadata(ExtInst, *Phi);

    Phi->replaceAllUsesWith(Ext);
    Phi->eraseFromParent();

    for (Instruction *Cast : DeadCasts)
      if (Cast->use_empty())
        Cast->eraseFromParent();

    Changed = true;
  }

  return Changed;
}

bool isLowHalfWidenShuffle(Value *V, FixedVectorType *SubVecTy,
                           Value *&SubVec) {
  auto *Shuffle = dyn_cast<ShuffleVectorInst>(V);
  if (!Shuffle ||
      Shuffle->getType() != FixedVectorType::get(SubVecTy->getElementType(), 8))
    return false;
  if (Shuffle->getOperand(0)->getType() != SubVecTy)
    return false;

  ArrayRef<int> Mask = Shuffle->getShuffleMask();
  if (Mask.size() != 8)
    return false;
  for (unsigned I = 0; I < 4; ++I)
    if (Mask[I] != static_cast<int>(I))
      return false;
  for (unsigned I = 4; I < 8; ++I)
    if (Mask[I] != PoisonMaskElem)
      return false;

  SubVec = Shuffle->getOperand(0);
  return true;
}

Value *getLowHalfConstant(Constant *C, FixedVectorType *SubVecTy) {
  if (isa<PoisonValue>(C))
    return PoisonValue::get(SubVecTy);
  if (C->isNullValue())
    return Constant::getNullValue(SubVecTy);

  SmallVector<Constant *, 4> LowElts;
  for (unsigned I = 0; I < 8; ++I) {
    auto *Elt = C->getAggregateElement(I);
    if (!Elt)
      return nullptr;
    if (I < 4) {
      LowElts.push_back(cast<Constant>(Elt));
      continue;
    }
    if (!isa<UndefValue>(Elt) && !isa<PoisonValue>(Elt))
      return nullptr;
  }
  return ConstantVector::get(LowElts);
}

bool onlyUsesLowHalf(Value *V) {
  auto UsesOnlyLowHalf = [&](ShuffleVectorInst *Shuffle) {
    auto *VecTy = cast<FixedVectorType>(V->getType());
    unsigned NumElts = VecTy->getNumElements();
    for (int MaskElt : Shuffle->getShuffleMask()) {
      if (MaskElt == PoisonMaskElem)
        continue;
      if (Shuffle->getOperand(0) == V && MaskElt < static_cast<int>(NumElts) &&
          MaskElt >= 4)
        return false;
      if (Shuffle->getOperand(1) == V && MaskElt >= static_cast<int>(NumElts) &&
          MaskElt - static_cast<int>(NumElts) >= 4)
        return false;
    }
    return true;
  };

  for (User *U : V->users()) {
    if (auto *Shuffle = dyn_cast<ShuffleVectorInst>(U)) {
      if (!UsesOnlyLowHalf(Shuffle))
        return false;
      continue;
    }
    auto *Extract = dyn_cast<ExtractElementInst>(U);
    if (!Extract)
      return false;
    auto *Idx = dyn_cast<ConstantInt>(Extract->getIndexOperand());
    if (!Idx || Idx->getValue().uge(4))
      return false;
  }
  return true;
}

bool repairLowHalfWidenedPhiCanonicalization(Function &F) {
  SmallVector<WeakTrackingVH, 8> Worklist;
  for (BasicBlock &BB : F) {
    for (PHINode &Phi : BB.phis())
      Worklist.push_back(&Phi);
  }

  bool Changed = false;
  for (WeakTrackingVH &Handle : Worklist) {
    auto *Phi = dyn_cast_or_null<PHINode>(Handle);
    if (!Phi)
      continue;
    auto *VecTy = dyn_cast<FixedVectorType>(Phi->getType());
    if (!VecTy || VecTy->getNumElements() != 8)
      continue;
    Type *EltTy = VecTy->getElementType();
    if (!EltTy->isFloatTy())
      continue;
    if (!onlyUsesLowHalf(Phi))
      continue;

    auto *SubVecTy = FixedVectorType::get(EltTy, 4);
    SmallVector<Value *, 4> SubIncomings;
    SubIncomings.reserve(Phi->getNumIncomingValues());

    bool CanRepair = true;
    for (Value *Incoming : Phi->incoming_values()) {
      Value *SubIncoming = nullptr;
      if (isLowHalfWidenShuffle(Incoming, SubVecTy, SubIncoming)) {
        SubIncomings.push_back(SubIncoming);
        continue;
      }
      auto *C = dyn_cast<Constant>(Incoming);
      if (C && (SubIncoming = getLowHalfConstant(C, SubVecTy))) {
        SubIncomings.push_back(SubIncoming);
        continue;
      }
      CanRepair = false;
      break;
    }
    if (!CanRepair)
      continue;

    BasicBlock::iterator InsertPt = Phi->getParent()->getFirstInsertionPt();
    if (InsertPt == Phi->getParent()->end())
      continue;

    PHINode *SubPhi =
        PHINode::Create(SubVecTy, Phi->getNumIncomingValues(),
                        Phi->getName() + ".lo", Phi->getIterator());
    copyIntermediateLocation(SubPhi, *Phi);
    for (unsigned I = 0, E = Phi->getNumIncomingValues(); I != E; ++I)
      SubPhi->addIncoming(SubIncomings[I], Phi->getIncomingBlock(I));

    IRBuilder<> Builder(Phi->getParent(), InsertPt);
    SmallVector<int, 8> Mask = {0,
                                1,
                                2,
                                3,
                                PoisonMaskElem,
                                PoisonMaskElem,
                                PoisonMaskElem,
                                PoisonMaskElem};
    Value *Widened = Builder.CreateShuffleVector(SubPhi, Mask, Phi->getName());
    if (auto *WidenedInst = dyn_cast<Instruction>(Widened))
      copyReplacementMetadata(WidenedInst, *Phi);
    Phi->replaceAllUsesWith(Widened);
    RecursivelyDeleteTriviallyDeadInstructions(Phi);
    Changed = true;
  }

  return Changed;
}

Value *getLowHalfWidenSource(Value *V) {
  auto *Shuffle = dyn_cast<ShuffleVectorInst>(V);
  if (!Shuffle)
    return nullptr;
  auto *WideTy = dyn_cast<FixedVectorType>(Shuffle->getType());
  auto *SubTy = dyn_cast<FixedVectorType>(Shuffle->getOperand(0)->getType());
  if (!WideTy || !SubTy || WideTy->getNumElements() != 8 ||
      SubTy->getNumElements() != 4 ||
      WideTy->getElementType() != SubTy->getElementType())
    return nullptr;

  ArrayRef<int> Mask = Shuffle->getShuffleMask();
  if (Mask.size() != 8)
    return nullptr;
  for (unsigned I = 0; I < 4; ++I)
    if (Mask[I] != static_cast<int>(I))
      return nullptr;
  for (unsigned I = 4; I < 8; ++I)
    if (Mask[I] != PoisonMaskElem)
      return nullptr;
  return Shuffle->getOperand(0);
}

bool combineWidenedLowHalfShuffles(Function &F) {
  SmallVector<WeakTrackingVH, 8> Worklist;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB)
      if (auto *Shuffle = dyn_cast<ShuffleVectorInst>(&I))
        Worklist.push_back(Shuffle);
  }

  bool Changed = false;
  for (WeakTrackingVH &Handle : Worklist) {
    auto *Shuffle = dyn_cast_or_null<ShuffleVectorInst>(Handle);
    if (!Shuffle)
      continue;
    Value *Lo = getLowHalfWidenSource(Shuffle->getOperand(0));
    Value *Hi = getLowHalfWidenSource(Shuffle->getOperand(1));
    if (!Lo || !Hi)
      continue;

    ArrayRef<int> Mask = Shuffle->getShuffleMask();
    if (Mask.size() != 8)
      continue;
    bool CanCombine = true;
    SmallVector<int, 8> NewMask;
    NewMask.reserve(8);
    for (int MaskElt : Mask) {
      if (MaskElt == PoisonMaskElem) {
        NewMask.push_back(PoisonMaskElem);
        continue;
      }
      if (MaskElt >= 0 && MaskElt < 4) {
        NewMask.push_back(MaskElt);
        continue;
      }
      if (MaskElt >= 8 && MaskElt < 12) {
        NewMask.push_back(MaskElt - 4);
        continue;
      }
      CanCombine = false;
      break;
    }
    if (!CanCombine)
      continue;

    IRBuilder<> Builder(Shuffle);
    Builder.SetCurrentDebugLocation(Shuffle->getDebugLoc());
    Value *Replacement = Builder.CreateShuffleVector(Lo, Hi, NewMask);
    if (auto *Inst = dyn_cast<Instruction>(Replacement))
      copyReplacementMetadata(Inst, *Shuffle);
    Shuffle->replaceAllUsesWith(Replacement);
    RecursivelyDeleteTriviallyDeadInstructions(Shuffle);
    Changed = true;
  }

  return Changed;
}

} // namespace

PreservedAnalyses MTGPUPostSLPRepairPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  bool Changed = repairFPExtPhiCanonicalization(F);
  Changed |= repairLowHalfWidenedPhiCanonicalization(F);
  Changed |= combineWidenedLowHalfShuffles(F);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
