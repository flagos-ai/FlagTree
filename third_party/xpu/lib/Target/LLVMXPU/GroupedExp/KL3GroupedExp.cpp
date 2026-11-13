#include "triton/Target/LLVMIR/KL3GroupedExp.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace {
constexpr unsigned Lanes = 16;
constexpr unsigned MaxPackets = 4;
constexpr Align PacketAlign = Align::Constant<64>();

IntrinsicInst *matchExp(Instruction &inst, FixedVectorType *packet) {
  auto *call = dyn_cast<IntrinsicInst>(&inst);
  if (!call || call->getIntrinsicID() != Intrinsic::exp ||
      call->getType() != packet || call->isMustTailCall() ||
      call->hasOperandBundles())
    return nullptr;
  return call;
}

// The pointer must remain live while all 16 scalar registers are overwritten.
// Each packet makes one round trip between the vector file and core-private LM.
void lowerBatch(ArrayRef<IntrinsicInst *> calls, AllocaInst *scratch) {
  IRBuilder<> builder(calls.front());
  Type *packet = calls.front()->getType();
  SmallVector<Value *, MaxPackets> slots;
  std::string instructions;
  raw_string_ostream code(instructions);
  code << "mfence { lm }\n";
  for (auto [i, call] : enumerate(calls)) {
    auto *slot = builder.CreateConstGEP1_32(packet, scratch, i);
    slots.push_back(slot);
    builder.CreateAlignedStore(call->getArgOperand(0), slot, PacketAlign, true);
    for (unsigned lane = 0; lane < Lanes; ++lane)
      code << "loadli.w $" << lane << ", " << (i * Lanes + lane) * 4
           << "($" << Lanes << ")\n";
    for (unsigned lane = 0; lane < Lanes; ++lane)
      code << "exp.f.rn $" << lane << ", $" << lane << "\n";
    for (unsigned lane = 0; lane < Lanes; ++lane)
      code << "storeli.w $" << lane << ", " << (i * Lanes + lane) * 4
           << "($" << Lanes << ")\n";
  }
  code << "mfence { lm }";

  SmallVector<Type *, Lanes> outputs(Lanes, builder.getInt32Ty());
  auto *resultTy = StructType::get(builder.getContext(), outputs);
  auto *asmTy = FunctionType::get(resultTy, {scratch->getType()}, false);
  std::string constraints;
  for (unsigned lane = 0; lane < Lanes; ++lane)
    constraints += "=&r,";
  constraints += "r,~{memory}";
  auto *grouped = InlineAsm::get(asmTy, instructions, constraints, true);
  builder.CreateCall(grouped, {scratch});
  for (auto [call, slot] : zip(calls, slots)) {
    builder.SetCurrentDebugLocation(call->getDebugLoc());
    auto *loaded = builder.CreateAlignedLoad(packet, slot, PacketAlign, true);
    call->replaceAllUsesWith(loaded);
  }
  for (auto *call : calls)
    call->eraseFromParent();
}

bool lowerFunction(Function &function, FixedVectorType *packet) {
  AllocaInst *scratch = nullptr;
  for (auto &block : function) {
    auto it = block.begin();
    while (it != block.end()) {
      auto *first = matchExp(*it++, packet);
      if (!first)
        continue;
      SmallVector<IntrinsicInst *, MaxPackets> batch{first};
      // Only group consecutive, independent calls. A dependent exp must see
      // the first group's reloaded result and execute in the next group.
      while (it != block.end() && batch.size() < MaxPackets) {
        auto *next = matchExp(*it, packet);
        if (!next || is_contained(batch, next->getArgOperand(0)))
          break;
        batch.push_back(next);
        ++it;
      }
      if (!scratch) {
        IRBuilder<> entry(&function.getEntryBlock(),
                          function.getEntryBlock().getFirstInsertionPt());
        scratch = entry.CreateAlloca(packet, entry.getInt32(MaxPackets),
                                     "kl3.exp.scratch");
        scratch->setAlignment(PacketAlign);
      }
      lowerBatch(batch, scratch);
    }
  }
  return scratch != nullptr;
}
} // namespace

PreservedAnalyses KL3GroupedExpPass::run(Module &module, ModuleAnalysisManager &) {
  auto *exp = module.getFunction("llvm.exp.v16f32");
  if (!exp || exp->use_empty() ||
      Triple(module.getTargetTriple()).getArchName() != "xpu3")
    return PreservedAnalyses::all();
  auto *packet =
      FixedVectorType::get(Type::getFloatTy(module.getContext()), Lanes);
  bool changed = false;
  for (auto &function : module) {
    if (function.isDeclaration() ||
        (function.hasFnAttribute("target-features") &&
         function.getFnAttribute("target-features")
             .getValueAsString()
             .contains("+cdnn")))
      continue;
    changed |= lowerFunction(function, packet);
  }
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
