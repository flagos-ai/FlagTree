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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace proton;

TEST(DecoderTest, Decode) {
  std::vector<uint8_t> testData = {0x78, 0x56, 0x34, 0x12, 0x01, 0x00,
                                   0x00, 0x80, 0xFF, 0xFF, 0xFF, 0xFF};

  auto buf = ByteSpan(testData.data(), testData.size());
  auto decoder = EntryDecoder(buf);
  auto entry1 = decoder.decode<I32Entry>();
  EXPECT_EQ(entry1->value, 0x12345678);
  auto entry2 = decoder.decode<CycleEntry>();
  EXPECT_EQ(entry2->isStart, false);
  EXPECT_EQ(entry2->scopeId, 0);
  EXPECT_EQ(entry2->cycle, 8589934591);
}
