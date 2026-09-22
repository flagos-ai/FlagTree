// REQUIRES: flagtree-common-ir
// RUN: triton-opt %s --split-input-file --convert-common-ir-to-ttgir --verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @partial_copy(%src: tensor<8x!tt.ptr<f32>>) {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    // expected-error@+1 {{requires a full-buffer tensor with matching shape and element type}}
    tile.copy %src -> %buf : tensor<8x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @copy_dtype(%src: tensor<16x!tt.ptr<i32>>) {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    // expected-error@+1 {{requires a full-buffer tensor with matching shape and element type}}
    tile.copy %src -> %buf : tensor<16x!tt.ptr<i32>>, !tile.buf<[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @copy_nonglobal(%src: tensor<16x!tt.ptr<f32, 3>>) {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    // expected-error@+1 {{GPU tile.copy requires a tensor of global pointers}}
    tile.copy %src -> %buf : tensor<16x!tt.ptr<f32, 3>>, !tile.buf<[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @copy_scalar(%src: !tt.ptr<f32>) {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    // expected-error@+1 {{GPU tile.copy requires a tensor of global pointers}}
    tile.copy %src -> %buf : !tt.ptr<f32>, !tile.buf<[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @copy_transpose(%src: tensor<16x!tt.ptr<f32>>) {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    // expected-error@+1 {{GPU tile.copy supports only plain ND copies}}
    tile.copy %src -> %buf {transpose} : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @copy_nz(%src: tensor<16x!tt.ptr<f32>>) {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    // expected-error@+1 {{GPU tile.copy supports only plain ND copies}}
    tile.copy %src -> %buf {src_layout = 1 : i64} : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @stride() {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    // expected-error@+1 {{GPU tile.subview requires unit strides}}
    %view = tile.subview %buf[%c0] [[8]] [[2]] {tle.gpu_layout = #shared} : <[16], f32, shared> -> <[8], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @sizes() {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    // expected-error@+1 {{tile.subview sizes must match its result shape}}
    %view = tile.subview %buf[%c0] [[4]] [[1]] {tle.gpu_layout = #shared} : <[16], f32, shared> -> <[8], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @offset_bounds() {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    // expected-error@+1 {{tile.subview offset is outside the source buffer}}
    %view = tile.subview %buf[%c16] [[8]] [[1]] {tle.gpu_layout = #shared} : <[16], f32, shared> -> <[8], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @dynamic_same_rank(%idx: index) {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    // expected-error@+1 {{same-rank tile.subview currently requires static offsets}}
    %view = tile.subview %buf[%idx] [[8]] [[1]] {tle.gpu_layout = #shared} : <[16], f32, shared> -> <[8], f32, shared>
    tt.return
  }
}

// -----

#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @trailing_offset() {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared2} : <[2, 16], f32, shared>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error@+1 {{rank-reducing tile.subview requires zero trailing offsets}}
    %view = tile.subview %buf[%c0, %c1] [[16]] [[1]] {tle.gpu_layout = #shared} : <[2, 16], f32, shared> -> <[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @bridge_mismatch() {
    %buf = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    // expected-error@+1 {{CommonIR buffer bridge descriptor type does not match the converted buffer}}
    %desc = builtin.unrealized_conversion_cast %buf : !tile.buf<[16], f32, shared> to !ttg.memdesc<8xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @unsupported_storage() {
    // expected-error@+1 {{GPU TileIR conversion currently requires #tile.shared}}
    %buf = tile.alloc {space = 8 : i64, tle.gpu_layout = #shared} : <[16], f32, local>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @missing_layout() {
    // expected-error@+1 {{is missing the preserved tle.gpu_layout attribute}}
    %buf = tile.alloc {space = 7 : i64} : <[16], f32, shared>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @unsupported_set_flag() {
    // expected-error@+1 {{GPU CommonIR conversion does not support tile engine/event synchronization}}
    "tile.set_flag"() {producer = 1 : i64, consumer = 6 : i64, event = 0 : i64} : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @unsupported_wait_flag() {
    // expected-error@+1 {{GPU CommonIR conversion does not support tile engine/event synchronization}}
    "tile.wait_flag"() {producer = 1 : i64, consumer = 6 : i64, event = 0 : i64} : () -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @unsupported_pipe_barrier() {
    // expected-error@+1 {{GPU CommonIR conversion does not support tile engine/event synchronization}}
    "tile.pipe_barrier"() {pipe = 1 : i64} : () -> ()
    tt.return
  }
}
