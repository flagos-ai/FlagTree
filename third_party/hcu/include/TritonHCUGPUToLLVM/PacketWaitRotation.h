#ifndef TRITON_HCU_PACKET_WAIT_ROTATION_H
#define TRITON_HCU_PACKET_WAIT_ROTATION_H
#include <string>
namespace llvm { class Module; }
namespace flagtree::hcu {
struct WaitRotationOptions {
  // Exact launch ABI supplied by metadata/host contract, never inferred from
  // the flat-work-group-size upper bound alone.
  unsigned threads = 0;
  unsigned sharedBytes = 0;
  unsigned modelIterations = 3;
};
struct WaitRotationStats {
  unsigned considered = 0, rotated = 0, groups = 0, modelEvents = 0;
  std::string reason;
};
WaitRotationStats rotatePacketWaits(llvm::Module &, const std::string &arch,
                                   const WaitRotationOptions &);
}
#endif
