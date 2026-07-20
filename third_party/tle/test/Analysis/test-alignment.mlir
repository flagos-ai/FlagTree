/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// RUN: triton-opt %s -test-print-alignment -split-input-file -verify-diagnostics=only-expected -o /dev/null

// TLE-specific axis-info coverage for remote pointer propagation.
tt.func @tle_remote_pointers_axis_info(%arg0: !tt.ptr<f16, 3> {tt.divisibility = 16 : i32}) {
  // expected-remark @below {{contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>}}
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [16], constancy = [128], constant_value = <none>}}
  %1 = tt.splat %arg0 : !tt.ptr<f16, 3> -> tensor<128x!tt.ptr<f16, 3>>
  // expected-remark @below {{contiguity = [128], divisibility = [16], constancy = [1], constant_value = <none>}}
  %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f16, 3>>, tensor<128xi32>
  // expected-remark @below {{contiguity = [1], divisibility = [4611686018427387904], constancy = [1], constant_value = 0}}
  %c0_i32 = arith.constant 0 : i32
  %3 = "tle.remote_pointers"(%2, %c0_i32) : (tensor<128x!tt.ptr<f16, 3>>, i32) -> tensor<128x!tt.ptr<f16, 7>>
  tt.return
}
