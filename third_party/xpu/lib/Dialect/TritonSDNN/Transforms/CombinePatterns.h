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

#ifndef TRITON_SDNN_COMBINE_PATTERNS_H
#define TRITON_SDNN_COMBINE_PATTERNS_H

namespace mlir {
namespace triton {
namespace sdnn {

void populateMMAScaleCombinePatterns(RewritePatternSet &patterns,
                                     uint32_t xpu_arch);
void populateCombinePatterns(RewritePatternSet &patterns, uint32_t xpu_arch);
void populatePostCombinePatterns(RewritePatternSet &patterns,
                                 uint32_t xpu_arch);
void populateMaskEWPatterns(RewritePatternSet &patterns, uint32_t xpu_arch);
void populateDMAPatterns(RewritePatternSet &patterns, uint32_t xpu_arch);
void polulateEwBufferSeperate(RewritePatternSet &patterns);

} // namespace sdnn
} // namespace triton
} // namespace mlir

#endif // TRITON_SDNN_COMBINE_PATTERNS_H
