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
