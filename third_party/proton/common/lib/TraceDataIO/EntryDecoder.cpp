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

#include "TraceDataIO/EntryDecoder.h"

using namespace proton;

std::ostream &operator<<(std::ostream &os, const EntryBase &obj) {
  obj.print(os);
  return os;
}

void I32Entry::print(std::ostream &os) const { os << value; }

template <> void proton::decodeFn<I32Entry>(ByteSpan &buffer, I32Entry &entry) {
  entry.value = buffer.readInt32();
}

void I64Entry::print(std::ostream &os) const { os << value; }

template <> void proton::decodeFn<I64Entry>(ByteSpan &buffer, I64Entry &entry) {
  entry.value = buffer.readInt64();
}

void CycleEntry::print(std::ostream &os) const {
  std::string prefix = isStart ? "S" : "E";
  os << prefix + std::to_string(scopeId) + "C" + std::to_string(cycle);
}

template <>
void proton::decodeFn<CycleEntry>(ByteSpan &buffer, CycleEntry &entry) {
  uint32_t tagClkUpper = buffer.readUInt32();
  entry.isStart = (tagClkUpper & 0x80000000) == 0;
  entry.scopeId = (tagClkUpper & 0x7F800000) >> 23;
  uint64_t clkLower = buffer.readUInt32();
  entry.cycle = static_cast<uint64_t>(tagClkUpper & 0x7FF) << 32 | clkLower;
}
