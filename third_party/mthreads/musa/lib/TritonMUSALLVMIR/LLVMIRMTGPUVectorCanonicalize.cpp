#include "TritonMUSALLVMIR/Passes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;
using namespace mlir::triton::musa;

namespace {

void copyVectorTransformMetadata(Instruction *Dst, const Instruction &Src,
                                 bool CopyFPMath = false) {
  SmallVector<unsigned, 4> Kinds = {LLVMContext::MD_dbg,
                                    LLVMContext::MD_annotation,
                                    LLVMContext::MD_nosanitize};
  if (CopyFPMath)
    Kinds.push_back(LLVMContext::MD_fpmath);
  Dst->copyMetadata(Src, Kinds);
}

bool hasCallSiteAttributes(const CallInst &Call) {
  AttributeList Attrs = Call.getAttributes();
  if (Attrs.getRetAttrs().hasAttributes() || Attrs.getFnAttrs().hasAttributes())
    return true;
  for (unsigned I = 0, E = Call.arg_size(); I != E; ++I)
    if (Attrs.getParamAttrs(I).hasAttributes())
      return true;
  return false;
}

bool canRewriteIntrinsicCall(const CallInst &Call) {
  const CallInst::TailCallKind TailKind = Call.getTailCallKind();
  return !Call.hasOperandBundles() && TailKind != CallInst::TCK_MustTail &&
         TailKind != CallInst::TCK_NoTail &&
         Call.getCallingConv() == CallingConv::C &&
         !hasCallSiteAttributes(Call);
}

Value *extractSubvector(IRBuilder<> &Builder, Value *Vec, unsigned Offset,
                        unsigned Width,
                        const Instruction *MetadataSrc = nullptr) {
  SmallVector<int, 16> Mask;
  Mask.reserve(Width);
  for (unsigned I = 0; I < Width; ++I)
    Mask.push_back(static_cast<int>(Offset + I));
  Value *Result = Builder.CreateShuffleVector(Vec, Mask);
  if (MetadataSrc)
    if (auto *Inst = dyn_cast<Instruction>(Result))
      copyVectorTransformMetadata(Inst, *MetadataSrc);
  return Result;
}

Value *insertSubvector(IRBuilder<> &Builder, Value *Base, Value *SubVec,
                       uint64_t Offset,
                       const Instruction *MetadataSrc = nullptr) {
  auto *BaseTy = cast<FixedVectorType>(Base->getType());
  auto *SubTy = cast<FixedVectorType>(SubVec->getType());

  if (SubTy->getNumElements() != BaseTy->getNumElements()) {
    SmallVector<int, 16> PadMask;
    PadMask.reserve(BaseTy->getNumElements());
    for (unsigned I = 0; I < BaseTy->getNumElements(); ++I)
      PadMask.push_back(I < SubTy->getNumElements() ? static_cast<int>(I)
                                                    : PoisonMaskElem);
    SubVec = Builder.CreateShuffleVector(SubVec, PadMask);
    if (MetadataSrc)
      if (auto *Inst = dyn_cast<Instruction>(SubVec))
        copyVectorTransformMetadata(Inst, *MetadataSrc);
  }

  SmallVector<int, 16> Mask;
  Mask.reserve(BaseTy->getNumElements());
  for (unsigned I = 0; I < BaseTy->getNumElements(); ++I) {
    if (I >= Offset && I < Offset + SubTy->getNumElements())
      Mask.push_back(static_cast<int>(BaseTy->getNumElements() + I - Offset));
    else
      Mask.push_back(static_cast<int>(I));
  }
  return Builder.CreateShuffleVector(Base, SubVec, Mask);
}

bool splitWideExp2Intrinsics(Function &F) {
  SmallVector<IntrinsicInst *, 8> Calls;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II || II->getIntrinsicID() != Intrinsic::exp2)
        continue;
      auto *VecTy = dyn_cast<FixedVectorType>(II->getType());
      if (!VecTy || !VecTy->getElementType()->isFloatTy() ||
          VecTy->getNumElements() != 8 || !canRewriteIntrinsicCall(*II))
        continue;
      Calls.push_back(II);
    }
  }

  bool Changed = false;
  for (IntrinsicInst *Call : Calls) {
    IRBuilder<> Builder(Call);
    Builder.SetCurrentDebugLocation(Call->getDebugLoc());

    auto *VecTy = cast<FixedVectorType>(Call->getType());
    auto *HalfTy = FixedVectorType::get(VecTy->getElementType(), 4);
    Value *Src = Call->getArgOperand(0);
    Value *LoSrc = extractSubvector(Builder, Src, 0, 4, Call);
    Value *HiSrc = extractSubvector(Builder, Src, 4, 4, Call);

    CallInst *Lo =
        Builder.CreateIntrinsic(Intrinsic::exp2, {HalfTy}, {LoSrc}, Call);
    copyVectorTransformMetadata(Lo, *Call, true);
    Lo->setFastMathFlags(Call->getFastMathFlags());
    CallInst *Hi =
        Builder.CreateIntrinsic(Intrinsic::exp2, {HalfTy}, {HiSrc}, Call);
    copyVectorTransformMetadata(Hi, *Call, true);
    Hi->setFastMathFlags(Call->getFastMathFlags());

    SmallVector<int, 16> MergeMask;
    MergeMask.reserve(VecTy->getNumElements());
    for (unsigned I = 0; I < VecTy->getNumElements(); ++I)
      MergeMask.push_back(static_cast<int>(I));
    Value *Replacement = Builder.CreateShuffleVector(Lo, Hi, MergeMask);
    if (auto *Inst = dyn_cast<Instruction>(Replacement))
      copyVectorTransformMetadata(Inst, *Call);

    Call->replaceAllUsesWith(Replacement);
    Call->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

bool lowerVectorInsertIntrinsics(Function &F) {
  SmallVector<IntrinsicInst *, 8> Calls;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Call = dyn_cast<IntrinsicInst>(&I);
      if (!Call || Call->getIntrinsicID() != Intrinsic::vector_insert)
        continue;
      if (!canRewriteIntrinsicCall(*Call))
        continue;
      Calls.push_back(Call);
    }
  }

  bool Changed = false;
  for (IntrinsicInst *Call : Calls) {
    auto *BaseTy = dyn_cast<FixedVectorType>(Call->getType());
    auto *SubTy = dyn_cast<FixedVectorType>(Call->getArgOperand(1)->getType());
    auto *Offset = dyn_cast<ConstantInt>(Call->getArgOperand(2));
    if (!BaseTy || !SubTy || !Offset)
      continue;
    unsigned BaseElements = BaseTy->getNumElements();
    unsigned SubElements = SubTy->getNumElements();
    if (SubElements > BaseElements)
      continue;
    APInt MaxOffset(Offset->getValue().getBitWidth(),
                    BaseElements - SubElements);
    if (Offset->getValue().ugt(MaxOffset))
      continue;
    uint64_t OffsetVal = Offset->getZExtValue();
    if (OffsetVal > BaseElements - SubElements)
      continue;

    IRBuilder<> Builder(Call);
    Builder.SetCurrentDebugLocation(Call->getDebugLoc());
    Value *Replacement =
        insertSubvector(Builder, Call->getArgOperand(0), Call->getArgOperand(1),
                        OffsetVal, Call);
    if (auto *Inst = dyn_cast<Instruction>(Replacement))
      copyVectorTransformMetadata(Inst, *Call);
    Call->replaceAllUsesWith(Replacement);
    Call->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

} // namespace

PreservedAnalyses
MTGPUCanonicalizeVectorIntrinsicsPass::run(Function &F,
                                           FunctionAnalysisManager &AM) {
  bool Changed = splitWideExp2Intrinsics(F);
  Changed |= lowerVectorInsertIntrinsics(F);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
