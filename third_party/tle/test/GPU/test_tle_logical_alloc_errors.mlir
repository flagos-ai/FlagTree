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
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -split-input-file -triton-tle-plan-logical-domains -verify-diagnostics

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// Logical payloads are attributes; memdesc matrix dimensions remain carriers.
// expected-error @+1 {{shape must have power-of-2 and non-zero dimensions}}
tt.func @logical_shape_is_not_a_carrier(%arg: !ttg.memdesc<2x80x256xf16, #shared, #ttg.shared_memory, mutable, 2x128x256>) {
  tt.return
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
tt.func @ordinary_subview_is_not_an_allocation() {
  // expected-error @+1 {{result shape and its alloc shape must match}}
  %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory, mutable, 128x128>
  tt.return
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
tt.func @wrong_rank2_carrier() {
  // expected-error @+1 {{logical candidate carrier type does not match its padded shape}}
  %buf = ttg.local_alloc {tle.logical_alloc_shape = array<i64: 80, 256>, tle.logical_non_power_axis = 0 : i32, tle.storage_plan = "candidate"} : () -> !ttg.memdesc<256x256xf16, #shared, #ttg.shared_memory, mutable>
  tt.return
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
tt.func @stage_count_must_not_be_padded() {
  // expected-error @+1 {{logical candidate carrier type does not match its padded shape}}
  %buf = ttg.local_alloc {tle.logical_alloc_shape = array<i64: 3, 80, 256>, tle.logical_non_power_axis = 1 : i32, tle.storage_plan = "candidate"} : () -> !ttg.memdesc<4x128x256xf16, #shared, #ttg.shared_memory, mutable>
  tt.return
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
tt.func @stage_axis_is_not_the_payload_axis() {
  // expected-error @+1 {{logical candidate requires a rank-2 matrix with an optional leading stage dimension}}
  %buf = ttg.local_alloc {tle.logical_alloc_shape = array<i64: 3, 80, 256>, tle.logical_non_power_axis = 0 : i32, tle.storage_plan = "candidate"} : () -> !ttg.memdesc<3x128x256xf16, #shared, #ttg.shared_memory, mutable>
  tt.return
}
