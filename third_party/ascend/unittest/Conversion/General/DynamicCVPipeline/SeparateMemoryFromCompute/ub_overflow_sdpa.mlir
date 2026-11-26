// RUN: triton-opt -split-input-file --ub-overflow-check %s | FileCheck %s
// RUN: triton-opt -split-input-file --ub-overflow-check --debug-only=UBOverflow %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEBUG

// The allocs below match the SDPA dump. With the mask double-buffered,
// alloc-only UB is 1318912 bits (161 KiB), so the mark is safe.

// CHECK-LABEL: func.func @alloc_only_keeps_mark
func.func @alloc_only_keeps_mark() {
  scope.scope : () -> () {
    %cbuf = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
    %a0 = memref.alloc() : memref<128x256xf32, #hivm.address_space<ub>>
    %a1 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    %a2 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %a3 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %a4 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %a5 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    // CHECK: %[[SAFE_MASK:.*]] = memref.alloc() : memref<128x256xi8>
    // CHECK-NEXT: annotation.mark %[[SAFE_MASK]] {hivm.multi_buffer = 2 : i32} : memref<128x256xi8>
    %mask = memref.alloc() : memref<128x256xi8>
    annotation.mark %mask {hivm.multi_buffer = 2 : i32} : memref<128x256xi8>
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// DEBUG: initial UB = 1318912 bits, max = 2031616 bits
// DEBUG: safe, no pruning needed

// -----

// The five tensor.empty ops add 1181696 bits (144.25 KiB).
// Total UB becomes 2500608 bits (305.25 KiB), so the mark must be pruned.

// CHECK-LABEL: func.func @tensor_sizes_prune_mark
func.func @tensor_sizes_prune_mark() {
  scope.scope : () -> () {
    %cbuf = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
    %a0 = memref.alloc() : memref<128x256xf32, #hivm.address_space<ub>>
    %a1 = memref.alloc() : memref<128x128xf32, #hivm.address_space<ub>>
    %a2 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %a3 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %a4 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %a5 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    // CHECK: %[[PRUNED_MASK:.*]] = memref.alloc() : memref<128x256xi8>
    // CHECK-NEXT: annotation.mark %[[PRUNED_MASK]] : memref<128x256xi8>
    %mask = memref.alloc() : memref<128x256xi8>
    annotation.mark %mask {hivm.multi_buffer = 2 : i32} : memref<128x256xi8>

    %t0 = tensor.empty() : tensor<128x128xf32>
    %t1 = tensor.empty() : tensor<128xf32>
    %t2 = tensor.empty() : tensor<128x256xf32>
    %t3 = tensor.empty() : tensor<128x256xi8>
    %t4 = tensor.empty() : tensor<16x128x16xf16>
    scope.return
  } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  return
}

// DEBUG: initial UB = 2500608 bits, max = 2031616 bits
// DEBUG: after deleting mark #1 (size 131072 bits): UB = 2369536 bits
