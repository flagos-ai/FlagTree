// RUN: triton-opt %s -split-input-file -inline -canonicalize -verify-each | FileCheck %s

module {
  // A callee promise belongs to each call operand, including when the caller
  // also reads the same pointer without that promise.
  // CHECK-LABEL: tt.func public @pointer_hints
  // CHECK-SAME: (%[[PTR:.*]]: !tt.ptr<i32>, %[[OTHER:.*]]: !tt.ptr<i32>)
  tt.func public @pointer_hints(%pointer: !tt.ptr<i32>, %other: !tt.ptr<i32>) {
    // CHECK: tt.load %[[PTR]]
    %unused = tt.load %pointer {isVolatile = true} : !tt.ptr<i32>
    // CHECK: %[[HINTED:.*]] = tt.bitcast %[[PTR]] {tt.divisibility = 256 : i32}
    // CHECK: tt.load %[[HINTED]]
    %first = tt.call @nested_pointer(%pointer) : (!tt.ptr<i32>) -> i32
    // CHECK: %[[OTHER_HINTED:.*]] = tt.bitcast %[[OTHER]] {tt.divisibility = 256 : i32}
    // CHECK: tt.load %[[OTHER_HINTED]]
    %second = tt.call @promised_pointer(%other) : (!tt.ptr<i32>) -> i32
    tt.return
  }
  tt.func private @nested_pointer(%pointer: !tt.ptr<i32>) -> i32 attributes {noinline = false} {
    %value = tt.call @promised_pointer(%pointer) : (!tt.ptr<i32>) -> i32
    tt.return %value : i32
  }
  tt.func private @promised_pointer(%pointer: !tt.ptr<i32> {tt.divisibility = 256 : i32}) -> i32 attributes {noinline = false} {
    %value = tt.load %pointer {isVolatile = true} : !tt.ptr<i32>
    tt.return %value : i32
  }
  // CHECK-NOT: tt.func private
}

// -----

module {
  // CHECK-LABEL: tt.func public @integer_hints
  // CHECK-SAME: (%[[VALUE:.*]]: i64, %[[OUTPUT:.*]]: !tt.ptr<i64>)
  tt.func public @integer_hints(%value: i64, %output: !tt.ptr<i64>) {
    // CHECK: %[[HINTED:.*]] = tt.bitcast %[[VALUE]] {tt.divisibility = 256 : i32}
    // CHECK: tt.store %[[OUTPUT]], %[[HINTED]]
    %result = tt.call @promised_integer(%value) : (i64) -> i64
    tt.store %output, %result : !tt.ptr<i64>
    tt.return
  }
  tt.func private @promised_integer(%value: i64 {tt.divisibility = 256 : i32}) -> i64 attributes {noinline = false} {
    tt.return %value : i64
  }
  // CHECK-NOT: tt.func private
}

// -----

module {
  // CHECK-LABEL: tt.func public @tensor_hints
  // CHECK-SAME: (%[[VALUE:.*]]: tensor<128xi32>, %[[OUTPUT:.*]]: tensor<128x!tt.ptr<i32>>)
  tt.func public @tensor_hints(%value: tensor<128xi32>, %output: tensor<128x!tt.ptr<i32>>) {
    // CHECK: %[[HINTED:.*]] = tt.bitcast %[[VALUE]] {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<8> : tensor<1xi32>, tt.divisibility = dense<8> : tensor<1xi32>}
    // CHECK: tt.store %[[OUTPUT]], %[[HINTED]]
    %result = tt.call @promised_tensor(%value) : (tensor<128xi32>) -> tensor<128xi32>
    tt.store %output, %result : tensor<128x!tt.ptr<i32>>
    tt.return
  }
  tt.func private @promised_tensor(%value: tensor<128xi32> {tt.divisibility = dense<8> : tensor<1xi32>, tt.contiguity = dense<8> : tensor<1xi32>, tt.constancy = dense<1> : tensor<1xi32>}) -> tensor<128xi32> attributes {noinline = false} {
    tt.return %value : tensor<128xi32>
  }
  // CHECK-NOT: tt.func private
}
