// RUN: triton-mlir-opt %s --coalesce-structured-store-sync

func.func private @store_ubuf_to_gm_1d_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32)

func.func private @custom_flagtree_stride_gather_float(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func private @mutate_data(memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>)

func.func private @unknown_effect()

func.func @disjoint(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @overlap(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 16)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @revisit(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @unknown_base(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %other to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @atomic(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 1 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @dynamic_mode(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %mode) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %mode) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %mode) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %mode) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @unaligned_ub(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 260 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @strided_gm(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [2] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [2] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [2] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [2] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[2], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @intervening_write(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      func.call @mutate_data(%srcd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @intervening_unknown(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      func.call @unknown_effect() : () -> ()
      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @other_pipeline(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_V>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @cross_pipeline(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}

func.func @final_wait(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]

      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
  }
  return
}

func.func @intervening_control(%gm: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %other: memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, %base: index, %mode: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %atomic = arith.constant 0 : i32
  %ci = arith.constant 0 : i64
  %ct = arith.constant 128 : i64
  %cd = arith.constant 256 : i64
  %cs = arith.constant 384 : i64
  %index = hivm.hir.pointer_cast(%ci) : memref<32xi32, #hivm.address_space<ub>>
  %scratch = hivm.hir.pointer_cast(%ct) : memref<32xi32, #hivm.address_space<ub>>
  %source = hivm.hir.pointer_cast(%cs) : memref<32xf32, #hivm.address_space<ub>>
  %dest = hivm.hir.pointer_cast(%cd) : memref<32xf32, #hivm.address_space<ub>>
  %indexd = memref.cast %index : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %scratchd = memref.cast %scratch : memref<32xi32, #hivm.address_space<ub>> to memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %load = memref.cast %source : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  %srcd = memref.cast %dest : memref<32xf32, #hivm.address_space<ub>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>
  scf.for %iv = %c0 to %c4 step %c1 {
    %cond = arith.cmpi eq, %iv, %c0 : index
    func.call @custom_flagtree_stride_gather_float(%load, %indexd, %srcd, %scratchd) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xi32, strided<[?], offset: ?>, #hivm.address_space<ub>>) -> ()
      %off0 = affine.apply affine_map<()[s0] -> (s0 + 0)>()[%base]
      %dst0 = memref.reinterpret_cast %gm to offset: [%off0], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc0 = memref.cast %dst0 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc0, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      scf.if %cond { scf.yield }
      %off1 = affine.apply affine_map<()[s0] -> (s0 + 32)>()[%base]
      %dst1 = memref.reinterpret_cast %gm to offset: [%off1], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc1 = memref.cast %dst1 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc1, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off2 = affine.apply affine_map<()[s0] -> (s0 + 64)>()[%base]
      %dst2 = memref.reinterpret_cast %gm to offset: [%off2], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc2 = memref.cast %dst2 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc2, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
      hivm.hir.pipe_barrier[<PIPE_MTE3>]
      %off3 = affine.apply affine_map<()[s0] -> (s0 + 96)>()[%base]
      %dst3 = memref.reinterpret_cast %gm to offset: [%off3], sizes: [32], strides: [1] : memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>> to memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
      %dstc3 = memref.cast %dst3 : memref<32xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>
      func.call @store_ubuf_to_gm_1d_float(%srcd, %dstc3, %atomic) : (memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<ub>>, memref<?xf32, strided<[?], offset: ?>, #hivm.address_space<gm>>, i32) -> ()
  }
  return
}
