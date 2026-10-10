// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=false" | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=15" | FileCheck %s

// Explicit layouts remain binding across if yields and loop-carried values.
// CHECK-DAG: #[[DST:[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [2],
#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "musa:31", tle.enable_encoding_rematerialization} {
  // CHECK-LABEL: tt.func @explicit_control_flow(
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: %[[THEN:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #[[DST]]} : {{.*}} -> tensor<256xf32, #[[DST]]>
  // CHECK: scf.yield %[[THEN]] : tensor<256xf32, #[[DST]]>
  // CHECK: } else {
  // CHECK: %[[ELSE:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #[[DST]]} : {{.*}} -> tensor<256xf32, #[[DST]]>
  // CHECK: scf.yield %[[ELSE]] : tensor<256xf32, #[[DST]]>
  // CHECK: scf.while
  // CHECK: scf.condition
  // CHECK: %[[BODY:.*]] = ttg.convert_layout {{.*}} {tle.explicit_encoding.0 = #[[DST]]} : {{.*}} -> tensor<256xf32, #[[DST]]>
  // CHECK: tt.return
  tt.func @explicit_control_flow(%input: tensor<256xf32, #src>, %other: tensor<256xf32, #src>, %cond: i1, %n: index) -> tensor<256xf32, #src> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %loop = scf.for %i = %c0 to %n step %c1 iter_args(%carry = %input) -> tensor<256xf32, #src> {
      %chosen = scf.if %cond -> tensor<256xf32, #dst> {
        %a = ttg.convert_layout %carry {tle.explicit_encoding.0 = #dst} : tensor<256xf32, #src> -> tensor<256xf32, #dst>
        scf.yield %a : tensor<256xf32, #dst>
      } else {
        %b = ttg.convert_layout %other {tle.explicit_encoding.0 = #dst} : tensor<256xf32, #src> -> tensor<256xf32, #dst>
        scf.yield %b : tensor<256xf32, #dst>
      }
      %back = ttg.convert_layout %chosen : tensor<256xf32, #dst> -> tensor<256xf32, #src>
      scf.yield %back : tensor<256xf32, #src>
    }
    %result = scf.while (%carry = %loop) : (tensor<256xf32, #src>) -> tensor<256xf32, #src> {
      scf.condition(%cond) %carry : tensor<256xf32, #src>
    } do {
    ^bb0(%carry: tensor<256xf32, #src>):
      %explicit = ttg.convert_layout %carry {tle.explicit_encoding.0 = #dst} : tensor<256xf32, #src> -> tensor<256xf32, #dst>
      %back = ttg.convert_layout %explicit : tensor<256xf32, #dst> -> tensor<256xf32, #src>
      scf.yield %back : tensor<256xf32, #src>
    }
    tt.return %result : tensor<256xf32, #src>
  }
}
