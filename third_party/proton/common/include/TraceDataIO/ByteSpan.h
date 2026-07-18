// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROTON_COMMON_BYTE_SPAN_H_
#define PROTON_COMMON_BYTE_SPAN_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace proton {

class BufferException : public std::runtime_error {
public:
  explicit BufferException(const std::string &message);
};

class ByteSpan {
public:
  ByteSpan(const uint8_t *data, size_t size);

  // Read methods
  uint8_t readUInt8();
  int8_t readInt8();
  uint16_t readUInt16();
  int16_t readInt16();
  uint32_t readUInt32();
  int32_t readInt32();
  uint64_t readUInt64();
  int64_t readInt64();

  // Buffer navigation
  void skip(size_t count);
  void seek(size_t position);
  size_t position() const { return pos; }
  size_t size() const { return dataSize; }
  size_t remaining() const { return dataSize - pos; }
  bool hasRemaining(size_t count = 0) const { return remaining() >= count; }

  // Data access
  const uint8_t *data() const { return dataPtr; }
  const uint8_t *currentData() const { return dataPtr + pos; }

private:
  const uint8_t *dataPtr; // Pointer to the underlying data
  size_t dataSize;        // Total size of the data
  size_t pos;             // Current read position

  // Helper method to check remaining bytes
  void checkRemaining(size_t required) const;
};

} // namespace proton

#endif // PROTON_COMMON_BYTE_SPAN_H_
