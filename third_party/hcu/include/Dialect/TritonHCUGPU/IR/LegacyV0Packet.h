#ifndef TRITON_HCU_DIALECT_IR_LEGACY_V0_PACKET_H_
#define TRITON_HCU_DIALECT_IR_LEGACY_V0_PACKET_H_

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

// Shared private packet contract. Keep lowering and wait-count selection in
// agreement without depending on a conversion pass or inventing a linear layout.
namespace mlir::triton::hcugpu {
using triton::gpu::HCUMlsSharedEncodingAttr;
struct LegacyV0Packet {
  unsigned k, n, warps, global;
};

inline std::optional<LegacyV0Packet> getLegacyV0Packet(Attribute encoding) {
  auto shared = dyn_cast<HCUMlsSharedEncodingAttr>(encoding);
  if (!shared || shared.getVersion() != 0 || shared.getOpIdx() != 1 ||
      shared.getElemBitWidth() != 16 || shared.getAlt2Kind() != 0 ||
      shared.getOrder() != ArrayRef<unsigned>({1, 0}))
    return std::nullopt;
  auto cta = shared.getCTALayout();
  if (!llvm::equal(cta.getCTAsPerCGA(), ArrayRef<unsigned>({1, 1})) ||
      !llvm::equal(cta.getCTASplitNum(), ArrayRef<unsigned>({1, 1})))
    return std::nullopt;
  if (shared.getMlsTile() == ArrayRef<unsigned>({32, 128}))
    return LegacyV0Packet{32, 128, 4, 1024};
  if (shared.getMlsTile() == ArrayRef<unsigned>({16, 256}))
    return LegacyV0Packet{16, 256, 8, 4096};
  return std::nullopt;
}

// Exact indexed-userA recipe, independent of the B packet selector above.
// The emitter issues two structured 16B-to-LDS calls for this complete packet.
inline bool isLegacyV0IndexedAPacket(Attribute attr) {
  auto e = dyn_cast<HCUMlsSharedEncodingAttr>(attr);
  return e && e.getVersion() == 0 && e.getOpIdx() == 0 &&
      e.getElemBitWidth() == 16 && e.getAlt2Kind() == 0 &&
      e.getMlsTile() == ArrayRef<unsigned>({128, 32}) &&
      e.getOrder() == ArrayRef<unsigned>({1, 0}) &&
      llvm::equal(e.getCTALayout().getCTAsPerCGA(), ArrayRef<unsigned>({1, 1})) &&
      llvm::equal(e.getCTALayout().getCTASplitNum(), ArrayRef<unsigned>({1, 1}));
}

} // namespace mlir::triton::hcugpu

#endif
