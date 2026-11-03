// Diagnostic counter experiment; never a proof of cross-wave LDS read-vs-DMA WAR.
// Fresh rotation proves the original full-wait schedule. Removing full waits
// retains only operand SSA readiness, checked again on actual native code.
#include "TritonHCUGPUToLLVM/PacketOperandOverlap.h"
#include "TritonHCUGPUToLLVM/PacketWaitRotation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <utility>

using namespace llvm;
namespace flagtree::hcu {
namespace {
StringRef callee(Instruction &i) {
  auto *c = dyn_cast<CallBase>(&i);
  auto *f = c ? c->getCalledFunction() : nullptr;
  return f ? f->getName() : StringRef();
}
bool wait(Instruction *i, unsigned &vm, unsigned &lgkm) {
  auto *c = dyn_cast_or_null<CallBase>(i);
  auto *f = c ? c->getCalledFunction() : nullptr;
  if (!f || f->getName() != "llvm.amdgcn.s.waitcnt" || c->arg_size() != 1)
    return false;
  auto *k = dyn_cast<ConstantInt>(c->getArgOperand(0));
  if (!k) return false;
  uint64_t x = k->getZExtValue();
  if (((x >> 4) & 7) != 7) return false;
  vm = unsigned((x & 15) | (((x >> 14) & 3) << 4));
  lgkm = unsigned((x >> 8) & 15);
  return true;
}
bool barrier(Instruction &i) {
  if (callee(i) == "llvm.amdgcn.s.barrier") return true;
  auto *c = dyn_cast<CallBase>(&i);
  auto *a = c ? dyn_cast<InlineAsm>(c->getCalledOperand()) : nullptr;
  return a && StringRef(a->getAsmString()).trim() == "s_barrier";
}
} // namespace
OperandOverlapResult preparePacketOperandOverlapDiagnostic(
    const std::string &source, const std::string &arch,
    const OperandOverlapOptions &options) {
  OperandOverlapResult result;
  result.source = source;
  auto reject = [&](const std::string &reason) {
    result.reason = reason;
    return result;
  };
  LLVMContext context;
  SMDiagnostic diagnostic;
  auto module = parseIR(MemoryBufferRef(source, "packet_operand_overlap"),
                        diagnostic, context);
  if (!module) return reject("invalid LLVM IR");
  // Reject malformed SSA before LoopInfo or the finite memory model sees it.
  std::string initialError;
  raw_string_ostream initialErrors(initialError);
  if (verifyModule(*module, &initialErrors))
    return reject("invalid input LLVM module: " + initialError);
  Function *kernel = nullptr;
  for (Function &f : *module) if (!f.isDeclaration()) {
    if (kernel) return reject("multiple defined functions");
    kernel = &f;
  }
  if (!kernel) return reject("no kernel");
  // Imported attributes are requests for native validation, never model proof.
  // Already scheduled sources are unchanged rather than rotated a second time.
  for (const char *attr : {"hcu.packet.wait.rotation", "hcu.packet.post_mma_wait",
                           "hcu.packet.operand_overlap.experimental", "hcu.mma.phase.loop"})
    if (kernel->hasFnAttribute(attr)) return reject("already scheduled/imported IR");
  WaitRotationOptions rotationOptions;
  rotationOptions.threads = options.threads;
  rotationOptions.sharedBytes = options.sharedBytes;
  rotationOptions.modelIterations = options.modelIterations;
  auto rotation = rotatePacketWaits(*module, arch, rotationOptions);
  if (rotation.rotated != 1 || !rotation.groups)
    return reject("fresh full-wait rotation: " + rotation.reason);
  DominatorTree dt(*kernel);
  LoopInfo li(dt);
  auto loops = li.getLoopsInPreorder();
  if (loops.size() != 1 || loops[0]->getNumBlocks() != 1)
    return reject("fresh rotation did not preserve one self loop");
  BasicBlock *body = loops[0]->getHeader();
  SmallVector<Instruction *> selected;
  unsigned vmWaits = 0, barriers = 0, mmas = 0;
  for (Instruction &i : *body) {
    barriers += barrier(i);
    StringRef name = callee(i);
    if (name.starts_with("llvm.hcu.mmac.") || name.starts_with("llvm.amdgcn.mfma.")) {
      auto *c = cast<CallBase>(&i);
      if (c->mayReadOrWriteMemory() || c->mayHaveSideEffects())
        return reject("MMA is not pure");
      ++mmas;
    }
    if (name != "llvm.amdgcn.s.waitcnt") continue;
    unsigned vm, lgkm;
    if (!wait(&i, vm, lgkm)) return reject("unknown wait immediate");
    if (vm < 63 && lgkm == 15) { ++vmWaits; continue; }
    unsigned priorVM, priorLGKM;
    if (vm != 63 || lgkm != 0 ||
        !wait(i.getPrevNode(), priorVM, priorLGKM) || priorVM == 63 || priorLGKM != 15)
      return reject("not one VM-only/full-LGKM pair per publication group");
    selected.push_back(&i);
  }
  // Check EVERY group before deleting anything. Counts derive from the fresh
  // structural/model proof, not from a shape, wave count or diagnostic SSA name.
  if (selected.size() != rotation.groups || vmWaits != rotation.groups ||
      barriers != rotation.groups || !mmas)
    return reject("incomplete publication group coverage");
  for (Instruction *i : selected) i->eraseFromParent();
  kernel->addFnAttr("hcu.packet.operand_overlap.experimental",
                    "CROSS_WAVE_WAR_UNPROVEN");
  std::string error;
  raw_string_ostream errors(error);
  if (verifyModule(*module, &errors)) return reject("operand-overlap LLVM verification: " + error);
  std::string output;
  raw_string_ostream stream(output);
  module->print(stream, nullptr);
  stream.flush();
  result.source = std::move(output);
  result.transformed = 1;
  result.groups = rotation.groups;
  result.removedWaits = selected.size();
  result.reason = "DIAGNOSTIC_ONLY_CROSS_WAVE_WAR_UNPROVEN";
  return result;
}
} // namespace flagtree::hcu
