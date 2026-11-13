// RUN: triton-opt --split-input-file %s -canonicalize -cse -o %t.control
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=8' -canonicalize -cse -o %t.template
// RUN: diff %t.control %t.template
// RUN: triton-opt --split-input-file --verify-each %s -graph-optimize='rule-mask=8 compile-mode=simt_only' -canonicalize -cse -o %t.simt
// RUN: diff %t.control %t.simt

// Unsupported inputs must remain unchanged, including all module attributes.
// scalar_select_preflight_fallback reaches private rewrite verification:
// a row-vector condition cannot select a higher-rank row tensor directly.
// A failed optional rewrite must not make valid input fail compilation.
// odd_grid_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 13 : i32} {
  tt.func @odd_grid_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// overflow_bound_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @overflow_bound_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %large = arith.constant 2147483647 : i32
    %upper = arith.addi %pid, %large : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// overlapping_store_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @overlapping_store_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %local : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// fp32_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @fp32_fallback(%A: !tt.ptr<f32>, %B: !tt.ptr<f32>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f32>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f32>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// unsafe_speculation_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @unsafe_speculation_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.addi %c2, %pid : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %den = arith.subi %upper, %iv : i32
      %quotient = arith.divsi %count, %den : i32
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// captured_combiner_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @captured_combiner_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %reduced = "tt.reduce"(%loop) <{axis = 0 : i32}> ({
    ^bb0(%reduce_a: f32, %reduce_b: f32):
      %sum = arith.addf %reduce_a, %reduce_b : f32
      %captured = arith.sitofp %count : i32 to f32
      %biased = arith.addf %sum, %captured : f32
      tt.reduce.return %biased : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %expanded = tt.expand_dims %reduced {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
    %output = tt.broadcast %expanded : tensor<1x16xf32> -> tensor<16x16xf32>
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %output : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// scalar_select_preflight_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @scalar_select_preflight_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %condition = arith.cmpi slt, %pid, %count : i32
    %selected = arith.select %condition, %loop, %loop : tensor<16x16xf32>
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %selected : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// row_rhs_load_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @row_rhs_load_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// loop_carried_rhs_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @loop_carried_rhs_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = arith.truncf %acc0 : tensor<16x16xf32> to tensor<16x16xf16>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// row_dependent_load_mask_fallback
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @row_dependent_load_mask_fallback(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %active = arith.cmpi slt, %pid, %iv : i32
      %mask = tt.splat %active : i1 -> tensor<16x16xi1>
      %zero = arith.constant dense<0.0> : tensor<16x16xf16>
      %b = tt.load %pb, %mask, %zero : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Reductions are unsupported, even with self-contained combiners.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @loop_reduction(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
    %reduced = "tt.reduce"(%dot0_0) <{axis = 1 : i32}> ({
    ^bb0(%reduce_a: f32, %reduce_b: f32):
      %sum = arith.addf %reduce_a, %reduce_b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<16x16xf32>) -> tensor<16xf32>
    %expanded = tt.expand_dims %reduced {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %output = tt.broadcast %expanded : tensor<16x1xf32> -> tensor<16x16xf32>
      scf.yield %output : tensor<16x16xf32>
    }

    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %loop : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @scalar_epilogue(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %flat = tt.reshape %loop : tensor<16x16xf32> -> tensor<256xf32>
    %reduced = "tt.reduce"(%flat) <{axis = 0 : i32}> ({
    ^bb0(%reduce_a: f32, %reduce_b: f32):
      %sum = arith.addf %reduce_a, %reduce_b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<256xf32>) -> f32
    %biased = arith.addf %reduced, %reduced : f32
    %output = tt.splat %biased : f32 -> tensor<16x16xf32>
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %output : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
// -----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, hacc.grid_num_tiles = 64 : i32} {
  tt.func @tuple_reduction(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %C: !tt.ptr<f32>, %count: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %stride = arith.constant 256 : i32
    %base = arith.muli %pid, %stride : i32
    %r = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %c = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %r2 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sixteen = arith.constant dense<16> : tensor<16x1xi32>
    %rs = arith.muli %r2, %sixteen : tensor<16x1xi32>
    %rr = tt.broadcast %rs : tensor<16x1xi32> -> tensor<16x16xi32>
    %cc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %local = arith.addi %rr, %cc : tensor<16x16xi32>
    %bases = tt.splat %base : i32 -> tensor<16x16xi32>
    %offset = arith.addi %bases, %local : tensor<16x16xi32>
    %ap = tt.splat %A : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pa = tt.addptr %ap, %offset : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>>
    %br = tt.expand_dims %c {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %bs = arith.constant dense<16> : tensor<16x1xi32>
    %bro = arith.muli %br, %bs : tensor<16x1xi32>
    %brr = tt.broadcast %bro : tensor<16x1xi32> -> tensor<16x16xi32>
    %bcc = tt.broadcast %c2d : tensor<1x16xi32> -> tensor<16x16xi32>
    %boff = arith.addi %brr, %bcc : tensor<16x16xi32>
    %bp = tt.splat %B : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
    %pb = tt.addptr %bp, %boff : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
    %upper = arith.constant 2 : i32
    %zero0 = arith.constant dense<0.0> : tensor<16x16xf32>
    %loop = scf.for %iv = %c0 to %upper step %c1 iter_args(%acc0 = %zero0) -> (tensor<16x16xf32>) : i32 {
      %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>>
      %dot0_0 = tt.dot %a, %b, %acc0 : tensor<16x16xf16> * tensor<16x16xf16> -> tensor<16x16xf32>
      scf.yield %dot0_0 : tensor<16x16xf32>
    }

    %pair:2 = "tt.reduce"(%loop, %loop) <{axis = 1 : i32}> ({
    ^bb0(%reduce_a: f32, %reduce_b: f32, %reduce_c: f32, %reduce_d: f32):
      %sum = arith.addf %reduce_a, %reduce_c : f32
      %max = arith.maximumf %reduce_b, %reduce_d : f32
      tt.reduce.return %sum, %max : f32, f32
    }) : (tensor<16x16xf32>, tensor<16x16xf32>) -> (tensor<16xf32>, tensor<16xf32>)
    %combined = arith.addf %pair#0, %pair#1 : tensor<16xf32>
    %expanded = tt.expand_dims %combined {axis = 1 : i32} : tensor<16xf32> -> tensor<16x1xf32>
    %output = tt.broadcast %expanded : tensor<16x1xf32> -> tensor<16x16xf32>
    %cp = tt.splat %C : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %pc = tt.addptr %cp, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %pc, %output : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
