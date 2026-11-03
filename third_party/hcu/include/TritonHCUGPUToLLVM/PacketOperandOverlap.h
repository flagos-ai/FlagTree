// Opt-in diagnostic only: native SSA readiness does not establish cross-wave WAR.
#ifndef TRITON_HCU_PACKET_OPERAND_OVERLAP_H
#define TRITON_HCU_PACKET_OPERAND_OVERLAP_H
#include <string>
namespace flagtree::hcu {
struct OperandOverlapOptions {
  unsigned threads = 0, sharedBytes = 0, modelIterations = 3;
};
struct OperandOverlapResult {
  // Transactional API: on every rejection this is byte-for-byte input source.
  std::string source, reason;
  unsigned transformed = 0, groups = 0, removedWaits = 0;
};
OperandOverlapResult preparePacketOperandOverlapDiagnostic(
    const std::string &source, const std::string &arch,
    const OperandOverlapOptions &);
}
#endif
