// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTON_UTILITY_NUMERIC_H_
#define PROTON_UTILITY_NUMERIC_H_

#include <cstddef>

namespace proton {

template <typename T> constexpr T nextPowerOfTwo(T value) {
  if (value < 1) {
    return 1;
  }
  --value; // Decrement to handle the case where value is already a power of two
  for (size_t i = 1; i < sizeof(T) * 8; i <<= 1) {
    value |= value >> i; // Propagate the highest set bit to the right
  }
  return value + 1; // Increment to get the next power of two
}

} // namespace proton

#endif // PROTON_UTILITY_NUMERIC_H_
