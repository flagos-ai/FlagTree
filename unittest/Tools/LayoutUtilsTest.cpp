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

#include "triton/Tools/LayoutUtils.h"

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Signals.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::triton {
namespace {

class LayoutUtilsTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LayoutUtilsTest, SquareSublayoutIsIdentity) {
  EXPECT_TRUE(squareSublayoutIsIdentity(
      LinearLayout::identity1D(4, S("in"), S("in")), {S("in")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(
      LinearLayout::identity1D(4, S("in"), S("in")), {}));

  LinearLayout l1(
      {{S("in1"), {{1, 1}, {2, 2}, {4, 4}}}, {S("in2"), {{2, 1}, {1, 2}}}},
      {{S("in1"), 8}, {S("in2"), 8}}, /*requireSurjective=*/false);
  EXPECT_TRUE(squareSublayoutIsIdentity(l1, {S("in1")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l1, {S("in2")}));

  LinearLayout l2 = LinearLayout::identity1D(4, S("in1"), S("in1")) *
                    LinearLayout::identity1D(8, S("in2"), S("in2")) *
                    LinearLayout({{S("in3"), {{1, 1, 1}}}},
                                 {{S("in1"), 2}, {S("in2"), 2}, {S("in3"), 2}},
                                 /*requireSurjective=*/false);
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in1")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in2")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l2, {S("in3")}));
  EXPECT_FALSE(squareSublayoutIsIdentity(l2, {S("in1"), S("in2")}));

  LinearLayout l3 = LinearLayout::identity1D(4, S("in1"), S("in1")) *
                    LinearLayout::identity1D(8, S("in2"), S("in2"));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in1")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in2")}));
  EXPECT_TRUE(squareSublayoutIsIdentity(l3, {S("in1"), S("in2")}));
}

} // namespace
} // namespace mlir::triton
