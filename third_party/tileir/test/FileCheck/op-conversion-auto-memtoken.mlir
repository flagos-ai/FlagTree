// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.experimental\$func(fuse-fma)),reconcile-unrealized-casts,auto-gen-memory-token{autogen-alias-memtoken=true})" | FileCheck %s

module {
  tt.func public @test_auto_memtoken_if_normal(%Cond: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %XTrue: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %XFalse: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %Ret: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %pid = tt.get_program_id x : i32
    %0 = arith.remsi %pid, %c2_i32 : i32
    %1 = arith.cmpi eq, %0, %c0_i32 : i32
    scf.if %1 {
      %2 = tt.load %XTrue : !tt.ptr<f32>
      tt.store %Ret, %2 : !tt.ptr<f32>
    } else {
      %2 = arith.cmpi eq, %0, %c1_i32 : i32
      scf.if %2 {
        %3 = tt.load %XFalse : !tt.ptr<f32>
        tt.store %Ret, %3 : !tt.ptr<f32>
      }
    }
    %4 = tt.load %Ret : !tt.ptr<f32>
    %5 = arith.addf %4, %4 : f32
    tt.store %Ret, %5 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_if_normal
// CHECK: %[[TOKEN0:.*]] = make_token : token
// CHECK: %[[TOKEN4:.*]] = if {{.*}} {
// CHECK-NOT:   {{.*}} = load_ptr_tko {{.*}}token=%{{.*}}
// CHECK:       %[[TOKEN1:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN0]] {{.*}}
// CHECK:       yield %[[TOKEN1]] : token
// CHECK: } else {
// CHECK:       %[[TOKEN3:.*]] = if {{.*}} {
// CHECK-NOT:       {{.*}} = load_ptr_tko {{.*}}token=%{{.*}}
// CHECK:           %[[TOKEN2:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN0]] {{.*}}
// CHECK:           yield %[[TOKEN2]] : token
// CHECK:       } else {
// CHECK:           yield %[[TOKEN0]] : token
// CHECK:       }
// CHECK:       yield %[[TOKEN3]] : token
// CHECK: }
// CHECK: {{.*}}, %[[TOKEN5:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN4]]
// CHECK: %[[TOKEN6:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN5]] {{.*}}

// -----


module {
  tt.func public @test_auto_memtoken_if_condensed1(%Cond: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %XTrue: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %XFalse: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %Ret: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %pid = tt.get_program_id x : i32
    %0 = arith.remsi %pid, %c2_i32 : i32
    %1 = arith.cmpi eq, %0, %c0_i32 : i32
    scf.if %1 {
      %2 = tt.load %XTrue : !tt.ptr<f32>
      tt.store %Ret, %2 : !tt.ptr<f32>
    } else {
      %2 = arith.cmpi eq, %0, %c1_i32 : i32
      scf.if %2 {
        %3 = tt.load %XFalse : !tt.ptr<f32>
        tt.store %Ret, %3 : !tt.ptr<f32>
      }
    }
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_if_condensed1
// CHECK: %[[TOKEN:.*]] = make_token : token
// CHECK: if {{.*}} {
// CHECK-NOT:   {{.*}} = load_ptr_tko {{.*}}token=%{{.*}}
// CHECK:       {{.*}} = store_ptr_tko {{.*}}token=%[[TOKEN]] {{.*}}
// CHECK-NOT:   yield {{.*}}
// CHECK: } else {
// CHECK        if {{.*}} {
// CHECK-NOT:       {{.*}} = load_ptr_tko {{.*}}token=%{{.*}}
// CHECK:           {{.*}} = store_ptr_tko {{.*}}token=%[[TOKEN]] {{.*}}
// CHECK-NOT:       yield {{.*}}
// CHECK-NOT:   } else {
// CHECK-NOT:       yield {{.*}}
// CHECK-NOT:   }
// CHECK-NOT:   yield {{.*}}
// CHECK: }

// -----


module {
  tt.func public @test_auto_memtoken_if_condensed2(%Cond: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %XTrue: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %XFalse: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %Ret: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %pid = tt.get_program_id x : i32
    %0 = arith.remsi %pid, %c2_i32 : i32
    %1 = arith.cmpi eq, %0, %c0_i32 : i32
    scf.if %1 {
      %2 = arith.cmpi eq, %0, %c1_i32 : i32
      scf.if %2 {
        %3 = tt.load %XFalse : !tt.ptr<f32>
        tt.store %Ret, %3 : !tt.ptr<f32>
      }
    } else {
      %2 = tt.load %XTrue : !tt.ptr<f32>
      tt.store %Ret, %2 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_if_condensed2
// CHECK: %[[TOKEN:.*]] = make_token : token
// CHECK: if {{.*}} {
// CHECK:       if {{.*}} {
// CHECK-NOT:       {{.*}} = load_ptr_tko {{.*}}token=%{{.*}}
// CHECK:           %[[TOKEN1:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN]] {{.*}}
// CHECK:           yield %[[TOKEN1]]
// CHECK:       } else {
// CHECK:           yield %[[TOKEN]]
// CHECK:       }
// CHECK-NOT:   yield {{.*}}
// CHECK: } else {
// CHECK-NOT:   {{.*}} = load_ptr_tko {{.*}}token=%{{.*}}
// CHECK:       {{.*}} = store_ptr_tko {{.*}}token=%[[TOKEN]] {{.*}}
// CHECK-NOT:   yield {{.*}}
// CHECK: }


// -----


module {
  tt.func public @test_auto_memtoken_for(%Out1: !tt.ptr<i64>, %Out2: !tt.ptr<i64>) attributes {noinline = false} {
    %c10000_i32 = arith.constant 10000 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<1> : tensor<128xi64>
    %c1_i32 = arith.constant 1 : i32
    %start = tt.elementwise_inline_asm "mov.u64 $0, %globaltimer;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %0 = tt.splat %Out1 : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>>
    %1 = tt.addptr %0, %off : tensor<128x!tt.ptr<i64>>, tensor<128xi32>
    scf.for %i = %c0_i32 to %c10000_i32 step %c1_i32  : i32 {
      %3 = tt.load %1 : tensor<128x!tt.ptr<i64>>
      %4 = arith.addi %3, %cst : tensor<128xi64>
      tt.store %1, %4 : tensor<128x!tt.ptr<i64>>
    }
    %end = tt.elementwise_inline_asm "mov.u64 $0, %globaltimer;" {constraints = "=l", packed_element = 1 : i32, pure = false} -> i64
    tt.store %Out2, %start : !tt.ptr<i64>
    %2 = tt.addptr %Out2, %c1_i32 : !tt.ptr<i64>, i32
    tt.store %2, %end : !tt.ptr<i64>
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_for
// CHECK: %[[TOKEN0:.*]] = make_token : token
// CHECK: %[[TOKEN:.*]] = for {{.*}} iter_values(%[[TOKEN1:.*]] = %[[TOKEN0]]) -> (token) {
// CHECK:     {{.*}}, %[[TOKEN2:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN1]]
// CHECK:     %[[TOKEN3:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN2]]
// CHECK:     continue %[[TOKEN3]] : token
// CHECK: }

// -----

module {
  tt.func public @test_auto_memtoken_nested_while(%data: !tt.ptr<f32>, %countPtr: !tt.ptr<i32>) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %count = tt.load %countPtr : !tt.ptr<i32>
      %count_0 = scf.while (%count_1 = %count) : (i32) -> i32 {
        %0 = arith.cmpi sgt, %count_1, %c0_i32 : i32
        scf.condition(%0) %count_1 : i32
      } do {
      ^bb0(%count_1: i32):
        %0 = tt.load %data : !tt.ptr<f32>
        %1 = arith.addf %0, %cst : f32
        tt.store %data, %1 : !tt.ptr<f32>
        %count_2 = arith.subi %count_1, %c2_i32 : i32
        scf.yield %count_2 : i32
      }
    }
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_nested_while
// CHECK: %[[TOKEN0:.*]] = make_token : token
// CHECK: %[[TOKEN:.*]] = for {{.*}} iter_values(%[[TOKEN1:.*]] = %[[TOKEN0]]) -> (token) {
// CHECK-NOT: {{.*}} = load_ptr_tko {{.*}}token={{.*}}
// CHECK:     %[[LOOP_RESULT:.*]]:2 = loop iter_values({{.*}}, %[[TOKEN2:.*]] = %[[TOKEN1]]) : {{.*}} {
// CHECK:       if {{.*}} {
// CHECK:         {{.*}}, %[[TOKEN3:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN2]]
// CHECK:         %[[TOKEN4:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN3]]
// CHECK:         continue {{.*}}, %[[TOKEN4]]
// CHECK:       } else {
// CHECK:         break {{.*}}, %[[TOKEN2]]
// CHECK:       }
// CHECK:       break {{.*}}, %[[TOKEN2]]
// CHECK:     }
// CHECK:     continue %[[LOOP_RESULT]]#1 : token
// CHECK: }


// -----

module {
  tt.func public @test_auto_memtoken_read_only(%X: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %Y: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %1 = tt.load %X : !tt.ptr<i32>
    %2 = tt.load %X : !tt.ptr<i32>
    %3 = tt.load %Y : !tt.ptr<i32>
    tt.store %Y, %3 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_read_only
// CHECK: %[[TOKEN0:.*]] = make_token : token
// CHECK: {{.*}}, %[[TOKEN1:.*]] = load_ptr_tko weak %{{.*}} : {{.*}}
// CHECK: {{.*}}, %[[TOKEN2:.*]] = load_ptr_tko weak %{{.*}} : {{.*}}
// CHECK: {{.*}}, %[[TOKEN3:.*]] = load_ptr_tko weak %{{.*}} token=%[[TOKEN0]] : {{.*}}
// CHECK: %[[TOKEN4:.*]] = store_ptr_tko {{.*}} token=%[[TOKEN3]] {{.*}}
