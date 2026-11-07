#ifndef TRITON_HCU_POST_MMA_WAIT_H
#define TRITON_HCU_POST_MMA_WAIT_H
#include <string>
namespace llvm { class Module; }
namespace flagtree::hcu {
struct PostMmaWaitOptions {
  // Exact launch ABI supplied by metadata/host contract, never inferred from
  // the flat-work-group-size upper bound alone.
  unsigned threads = 0;
  unsigned sharedBytes = 0;
  unsigned modelIterations = 3;
};
struct PostMmaWaitStats {
  unsigned considered = 0, transformed = 0, groups = 0, modelEvents = 0;
  unsigned tightenedVMWaits = 0, movedCopies = 0;
  std::string reason;
};
PostMmaWaitStats placePostMmaWaits(llvm::Module &, const std::string &arch,
                                   const PostMmaWaitOptions &);
}
#endif
