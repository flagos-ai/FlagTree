// RUN: triton-mlir-opt %s --hoist-structured-permutation-addresses --allow-unregistered-dialect


func.func private @custom_flagtree_stride_gather_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func private @consume(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func private @store_ubuf_to_gm_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32)
func.func private @load_gm_to_ubuf_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32)
func.func private @arange_1d_int32_t(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index)
func.func private @unknown_effect()
func.func private @mutate(memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func @packed_double(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @packed_single(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %ss = memref.cast %sa : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @unaligned_source(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 388 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @index_extra_read(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      func.call @store_ubuf_to_gm_1d_int32_t(%ii, %gm, %atomic0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    }

    return
}

func.func @physical_alias(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %alias = hivm.hir.pointer_cast(%ca) : memref<32xi32, #hivm.address_space<ub>>
      %alias_d = memref.cast %alias : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @store_ubuf_to_gm_1d_int32_t(%alias_d, %gm, %atomic0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    }

    return
}

func.func @unknown_alias(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      func.call @store_ubuf_to_gm_1d_int32_t(%unknown, %gm, %atomic0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    }

    return
}

func.func @index_write(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      func.call @mutate(%ii) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
    }

    return
}

func.func @after_loop_read(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }
    func.call @store_ubuf_to_gm_1d_int32_t(%ii, %gm, %atomic0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    return
}

func.func @strided_indices(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, strided<[2]>, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, strided<[2]>, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @unaligned_indices(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 4 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @unaligned_scratch(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 132 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @unaligned_destination(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 260 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @overlapping_tables(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 0 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @source_destination_alias(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 256 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @conditional_gather(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      scf.if %cond {
        func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      }

    }

    return
}

func.func @unknown_source(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%dynamic) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @outside_loop_read_before(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    func.call @store_ubuf_to_gm_1d_int32_t(%ii, %gm, %atomic0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @zero_iterations(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 0 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @one_iteration(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 1 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @two_iterations(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 2 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @dynamic_iterations(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %trip: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %trip step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @unmodeled_scalar_effect(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>

    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @unknown_effect() : () -> ()
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @before_loop_dma_write(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    func.call @load_gm_to_ubuf_1d_int32_t(%gm, %ii, %atomic0) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, i32) -> ()
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}

func.func @before_loop_unknown_effect(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    func.call @unknown_effect() : () -> ()
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}


func.func @before_loop_vector_init(%unknown: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, %dynamic: i64, %gm: memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %ci = arith.constant 0 : i64
    %ct = arith.constant 128 : i64
    %cd = arith.constant 256 : i64
    %cs = arith.constant 384 : i64
    %cb = arith.constant 512 : i64
    %atomic0 = arith.constant 0 : i32
    %ca = arith.constant 64 : i64
    %i = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
    %t = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
    %d = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
    %sa = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
    %sb = hivm.hir.pointer_cast(%cb) : memref<32xf32, #hivm.address_space<ub>>
    %ii = memref.cast %i : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %tt = memref.cast %t : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    %dd = memref.cast %d : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
    func.call @arange_1d_int32_t(%ii, %c0, %c1) : (memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, index, index) -> ()
    scf.for %iv = %c0 to %c4 step %c1 {
      %cond = arith.cmpi eq, %iv, %c0 : index
      %selected = arith.select %cond, %sa, %sb : memref<32xf32, #hivm.address_space<ub>>
      %ss = memref.cast %selected : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
      func.call @custom_flagtree_stride_gather_float(%ss, %ii, %dd, %tt) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()

    }

    return
}
