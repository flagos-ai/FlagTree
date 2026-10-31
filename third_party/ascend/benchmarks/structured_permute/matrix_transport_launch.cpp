#include <aclrtlaunch_matrix_transport_aiv.h>
#include <cstdint>

extern "C" uint32_t launch_matrix_transport(void *stream, void *x, void *y,
                                            void *indices, uint32_t transpose) {
  return ACLRT_LAUNCH_KERNEL(matrix_transport_aiv)(
      40, stream, static_cast<uint8_t *>(x), static_cast<uint8_t *>(y),
      static_cast<uint8_t *>(indices), transpose);
}
