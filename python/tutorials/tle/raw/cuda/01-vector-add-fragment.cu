#include <stdint.h>

__attribute__((device, always_inline)) void
add_fragment(float *x, int64_t x_size, float *y, int64_t y_size) {
  for (int64_t i = 0; i < x_size; i++) {
    x[i] += y[i];
  }
}
