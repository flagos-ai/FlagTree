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

#ifndef PROTON_COMMON_ENTRY_DECODER_H_
#define PROTON_COMMON_ENTRY_DECODER_H_

#include "ByteSpan.h"
#include <cstdint>
#include <iostream>
#include <memory>

namespace proton {

class EntryBase;

template <typename EntryT> void decodeFn(ByteSpan &buffer, EntryT &entry) {
  throw std::runtime_error("No decoder function is implemented");
}

class EntryDecoder {
private:
  ByteSpan &buf;

public:
  explicit EntryDecoder(ByteSpan &buffer) : buf(buffer) {}

  template <typename EntryT> std::shared_ptr<EntryT> decode() {
    auto entry = std::make_shared<EntryT>();
    decodeFn<EntryT>(buffer(), *entry);
    return entry;
  }

protected:
  // Protected accessor for the buffer
  ByteSpan &buffer() { return buf; }
};

struct EntryBase {
  virtual ~EntryBase() = default;

  virtual void print(std::ostream &os) const = 0;
};

std::ostream &operator<<(std::ostream &os, const EntryBase &obj);

struct I32Entry : public EntryBase {
  I32Entry() = default;

  void print(std::ostream &os) const override;

  int32_t value = 0;
};

template <> void decodeFn<I32Entry>(ByteSpan &buffer, I32Entry &entry);

struct I64Entry : public EntryBase {
  I64Entry() = default;

  void print(std::ostream &os) const override;

  int64_t value = 0;
};

template <> void decodeFn<I64Entry>(ByteSpan &buffer, I64Entry &entry);

struct CycleEntry : public EntryBase {
  CycleEntry() = default;

  void print(std::ostream &os) const override;

  uint64_t cycle = 0;
  bool isStart = true;
  int32_t scopeId = 0;
};

template <> void decodeFn<CycleEntry>(ByteSpan &buffer, CycleEntry &entry);

} // namespace proton

#endif // PROTON_COMMON_ENTRY_DECODER_H_
