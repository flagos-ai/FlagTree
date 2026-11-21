#ifndef TRITON_HCU_LLVM_MODULE_TRANSFORMS_H
#define TRITON_HCU_LLVM_MODULE_TRANSFORMS_H
#include "llvm/Support/Error.h"
#include <string>
namespace llvm {
class Module;
}
namespace mlir::triton::HCU {
struct ReadOnlyInterleaveStats {
  unsigned windows = 0, globalReads = 0, ldsReads = 0, fences = 0;
  unsigned pureDependents = 0;
  unsigned rejectedDependencies = 0, invalidFunctions = 0, iglpFunctions = 0;
};
struct TupledRegisterFenceStats {
  unsigned sites = 0, scalarLanes = 0, rejected = 0;
};
unsigned preserveLdsBarrierOrder(llvm::Module *module, const std::string &arch);
llvm::Expected<unsigned>
stripSynchronousBarrierMarkers(llvm::Module *module, const std::string &arch);
llvm::Expected<unsigned> lowerTaggedLocalBarriers(llvm::Module *module,
                                                  const std::string &arch);
unsigned preserveUniformResourceIndices(llvm::Module *module,
                                        const std::string &arch);
unsigned preserveBufferLoadByteOffsets(llvm::Module *module,
                                       const std::string &arch);
unsigned preferReadySharedLoads(llvm::Module *module, const std::string &arch);
unsigned annotateLdsObjectScopes(llvm::Module *module, const std::string &arch,
                                 unsigned waveSize);
unsigned preserveLdsDmaAliasScopes(llvm::Module *module,
                                   const std::string &arch);
ReadOnlyInterleaveStats interleaveGlobalLdsReads(llvm::Module *module,
                                                 const std::string &arch,
                                                 bool addFences);
llvm::Expected<TupledRegisterFenceStats>
tupleRegisterFences(llvm::Module *module, const std::string &arch);
struct NativeMemoryOptimizationResult {
  std::string source;
  unsigned factoredStores = 0, packedBf16Rounds = 0, combinedShifts = 0;
  std::string shiftedReason;
};
llvm::Expected<NativeMemoryOptimizationResult>
optimizeMemoryIR(const std::string &source, const std::string &arch,
                 bool factorStores, bool packRounds, bool combineShifts);
} // namespace mlir::triton::HCU
#endif
