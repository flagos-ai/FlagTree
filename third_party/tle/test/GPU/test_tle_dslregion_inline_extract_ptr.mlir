// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s --tle-dslregion-inline | FileCheck %s

module {
  llvm.func @_sink(!llvm.ptr)

  tt.func @k(%arg0: !tt.ptr<i32>) {
    %0 = "tle.dsl_region"(%arg0) ({
    ^bb0(%in: !tt.ptr<i32>):
      %p = "tle.extract_ptr"(%in) : (!tt.ptr<i32>) -> !llvm.ptr
      "tle.yield"(%p) : (!llvm.ptr) -> ()
    }) : (!tt.ptr<i32>) -> (!llvm.ptr)
    llvm.call @_sink(%0) : (!llvm.ptr) -> ()
    tt.return
  }
}

// CHECK-LABEL: tt.func @k(
// CHECK-NOT: tle.dsl_region
// CHECK: %[[P:.*]] = tle.extract_ptr
// CHECK: llvm.call @_sink(%[[P]]) : (!llvm.ptr) -> ()
