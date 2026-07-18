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

#include "TraceDataIO/ByteSpan.h"

using namespace proton;

ByteSpan::ByteSpan(const uint8_t *data, size_t size)
    : dataPtr(data), dataSize(size), pos(0) {
  if (data == nullptr && size > 0) {
    throw std::invalid_argument(
        "Data pointer cannot be null for non-zero size");
  }
}

void ByteSpan::checkRemaining(size_t required) const {
  if (remaining() < required) {
    throw BufferException("");
  }
}

uint8_t ByteSpan::readUInt8() {
  checkRemaining(1);
  return dataPtr[pos++];
}

int8_t ByteSpan::readInt8() { return static_cast<int8_t>(readUInt8()); }

uint16_t ByteSpan::readUInt16() {
  checkRemaining(2);
  uint16_t value = static_cast<uint16_t>(dataPtr[pos]) |
                   (static_cast<uint16_t>(dataPtr[pos + 1]) << 8);
  pos += 2;
  return value;
}

int16_t ByteSpan::readInt16() { return static_cast<int16_t>(readUInt16()); }

uint32_t ByteSpan::readUInt32() {
  checkRemaining(4);
  uint32_t value = static_cast<uint32_t>(dataPtr[pos]) |
                   (static_cast<uint32_t>(dataPtr[pos + 1]) << 8) |
                   (static_cast<uint32_t>(dataPtr[pos + 2]) << 16) |
                   (static_cast<uint32_t>(dataPtr[pos + 3]) << 24);
  pos += 4;
  return value;
}

int32_t ByteSpan::readInt32() { return static_cast<int32_t>(readUInt32()); }

uint64_t ByteSpan::readUInt64() {
  checkRemaining(8);
  uint64_t value = static_cast<uint64_t>(dataPtr[pos]) |
                   (static_cast<uint64_t>(dataPtr[pos + 1]) << 8) |
                   (static_cast<uint64_t>(dataPtr[pos + 2]) << 16) |
                   (static_cast<uint64_t>(dataPtr[pos + 3]) << 24) |
                   (static_cast<uint64_t>(dataPtr[pos + 4]) << 32) |
                   (static_cast<uint64_t>(dataPtr[pos + 5]) << 40) |
                   (static_cast<uint64_t>(dataPtr[pos + 6]) << 48) |
                   (static_cast<uint64_t>(dataPtr[pos + 7]) << 56);
  pos += 8;
  return value;
}

int64_t ByteSpan::readInt64() { return static_cast<int64_t>(readUInt64()); }

void ByteSpan::skip(size_t count) {
  checkRemaining(count);
  pos += count;
}

void ByteSpan::seek(size_t position) {
  if (position > dataSize) {
    throw BufferException("");
  }
  pos = position;
}

BufferException::BufferException(const std::string &message)
    : std::runtime_error(message) {}
