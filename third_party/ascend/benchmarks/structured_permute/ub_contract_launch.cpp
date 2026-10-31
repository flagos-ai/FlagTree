#include <aclrtlaunch_ub_contract_aiv.h>
#include <cstdint>
extern "C" uint32_t launch_ub_contract(void *stream, void *x, void *indices,
                                       void *a, void *b, void *config,
                                       uint32_t mode, uint32_t wordBytes) {
  return ACLRT_LAUNCH_KERNEL(ub_contract_aiv)(
      1, stream, static_cast<uint8_t *>(x), static_cast<uint8_t *>(indices),
      static_cast<uint8_t *>(a), static_cast<uint8_t *>(b),
      static_cast<uint8_t *>(config), mode, wordBytes);
}

extern "C" uint32_t launch_short_row_contract(void *stream, void *x, void *y,
                                              void *debug, void *config,
                                              uint32_t programs,
                                              uint32_t wordBytes) {
  uint32_t blocks = programs < 40 ? programs : 40;
  return ACLRT_LAUNCH_KERNEL(ub_contract_aiv)(
      blocks, stream, static_cast<uint8_t *>(x), static_cast<uint8_t *>(x),
      static_cast<uint8_t *>(y), static_cast<uint8_t *>(debug),
      static_cast<uint8_t *>(config), 4, wordBytes);
}

extern "C" uint32_t launch_short_row_load_contract(void *stream, void *x,
                                                   void *y, void *debug,
                                                   void *config,
                                                   uint32_t programs,
                                                   uint32_t wordBytes) {
  uint32_t blocks = programs < 40 ? programs : 40;
  return ACLRT_LAUNCH_KERNEL(ub_contract_aiv)(
      blocks, stream, static_cast<uint8_t *>(x), static_cast<uint8_t *>(x),
      static_cast<uint8_t *>(y), static_cast<uint8_t *>(debug),
      static_cast<uint8_t *>(config), 5, wordBytes);
}
