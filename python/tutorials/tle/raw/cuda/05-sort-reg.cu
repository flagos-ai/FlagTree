#include <stdint.h>
#include "./tle_raw.h"
#include "device_atomic_functions.h"

// Fine-grained replacement for Triton's tl.histogram. Loading, ordered-key
// conversion, digit extraction, and the final global store stay in Triton.
TLE_RAW_FUNC RadixHistogramDigits8x2048(
    tle_raw::TensorFragment<uint16_t> digits,
    tle_raw::TensorFragment<bool> mask,
    tle_raw::PlainBuffer1D<uint32_t> histogram
) {
    for(unsigned i = 0; i < digits.size; i++) {
        if(mask[i]) atomicAdd(&histogram[digits[i]], 1u);
    }
    __syncthreads();
}
