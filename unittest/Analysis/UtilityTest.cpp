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

#include "triton/Dialect/Triton/IR/Utility.h"

#include "llvm/Support/Signals.h"
#include <gtest/gtest.h>

namespace mlir {

TEST(Analysis, reorder) {
  SmallVector<int> shape({10, 20, 30});
  {
    SmallVector<unsigned> order({2, 1, 0});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 30);
    EXPECT_EQ(reordered[1], 20);
    EXPECT_EQ(reordered[2], 10);
  }
  {
    SmallVector<unsigned> order({1, 0, 2});
    auto reordered = triton::applyPermutation(shape, order);
    EXPECT_EQ(reordered[0], 20);
    EXPECT_EQ(reordered[1], 10);
    EXPECT_EQ(reordered[2], 30);
  }
}

} // namespace mlir

int main(int argc, char *argv[]) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
