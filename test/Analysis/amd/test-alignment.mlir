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

// RUN: triton-opt %s -test-print-amd-alignment -split-input-file -verify-diagnostics=only-expected -o /dev/null

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>

tt.func public @kernel(%arg0: tensor<256x64xf16, #mma> {tt.contiguity=256 : i32, tt.divisibility=6: i32, tt.constancy=1: i32}) {
  // expeted-remark @below {{contiguity = [128, 32], divisibility = [6, 6], constancy = [1, 1], constant_value = <none>}}
  %0 = amdg.extract_slice %arg0 [128, 32] : tensor<256x64xf16, #mma> to tensor<128x32xf16, #mma>
  tt.return
}
