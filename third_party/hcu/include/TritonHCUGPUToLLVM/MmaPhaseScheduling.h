#ifndef TRITON_HCU_MMA_PHASE_SCHEDULING_H
#define TRITON_HCU_MMA_PHASE_SCHEDULING_H
#include "llvm/ADT/StringRef.h"
namespace llvm { class Module; }
namespace flagtree::hcu {
struct MmaPhaseStats {
  unsigned considered = 0, versioned = 0, mmaCalls = 0;
};
// The caller must validate native VM/LGKM completion before using the result.
// Defined in the LLVM-only conversion library, with its LLVM RTTI settings.
MmaPhaseStats prepareMmaPhaseLoops(llvm::Module &module, llvm::StringRef arch,
                                  unsigned numWarps);
} // namespace flagtree::hcu
#endif
