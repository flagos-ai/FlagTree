// RUN: triton-opt %s -split-input-file -triton-tle-lower-pipe-to-nvws -verify-diagnostics

// Wait validates the reader during lifecycle lowering; release validates it
// during drain analysis. Both phases must use the same declaration contract.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wait_named_without_readers(%data: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %data {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{uses named reader named but pipe was created without readers}}
    %closed = tle.pipe.reader_wait %pipe, %data[%zero, %false] {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, reader_name = "named"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wait_missing_name(%data: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %data {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, readers = ["known"]} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{requires reader_name because pipe was created with explicit readers}}
    %closed = tle.pipe.reader_wait %pipe, %data[%zero, %false] {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wait_undeclared_name(%data: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %data {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, readers = ["known"]} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{uses undeclared pipe reader other}}
    %closed = tle.pipe.reader_wait %pipe, %data[%zero, %false] {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, reader_name = "other"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @release_named_without_readers(%data: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %data {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{uses named reader named but pipe was created without readers}}
    tle.pipe.reader_release %pipe, %data[%zero] {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, reader_name = "named"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @release_missing_name(%data: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %data {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, readers = ["known"]} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{requires reader_name because pipe was created with explicit readers}}
    tle.pipe.reader_release %pipe, %data[%zero] {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @release_undeclared_name(%data: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %data {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, readers = ["known"]} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    // expected-error @+1 {{uses undeclared pipe reader other}}
    tle.pipe.reader_release %pipe, %data[%zero] {capacity = 1 : i32, field_names = ["data"], scope = "cta", one_shot = true, reader_name = "other"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
