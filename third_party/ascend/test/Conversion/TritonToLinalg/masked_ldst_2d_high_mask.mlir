// RUN: triton-adapter-opt --triton-to-annotation --triton-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func @kernel_high_mask(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32,
  %arg3 : i32,
  %arg4 : i32,
  %arg5 : i32
  )
  {
    // Mimic a scenario where the raw pointer points to a buffer with dimension (1024, 1024)
    // in row-major, but the actual tensor size is (arg2, arg3).
    // We are trying to load a 128x256 sub-buffer starting at (2, 3).
    // The resulting memref:
    //  offset = 3074
    //  size[1] = 128
    //  size[0] = 256
    //  stride[0] = 1024
    //  stride[1] = 1
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    // horizontal index
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c2 = arith.constant 2 : i32
    %c2tensor = tt.splat %c2 : i32 -> tensor<128xi32>
    %offset2 = arith.addi %2, %c2tensor : tensor<128xi32>
    %3 = tt.expand_dims %offset2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : tensor<128x1xi32> -> tensor<128x256xi32>
    // vertical index
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %c3 = arith.constant 3 : i32
    %c3tensor = tt.splat %c3 : i32 -> tensor<256xi32>
    %offset5 = arith.addi %5, %c3tensor : tensor<256xi32>
    %c1024 = arith.constant 1024 : i32
    %c1024tensor = tt.splat %c1024 : i32 -> tensor<256xi32>
    %scale5 = arith.muli %offset5, %c1024tensor : tensor<256xi32>
    %6 = tt.expand_dims %scale5 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %7 = tt.broadcast %6 : tensor<1x256xi32> -> tensor<128x256xi32>
    // combined index
    %index = arith.addi %4, %7 : tensor<128x256xi32>
    %ldptr = tt.addptr %0, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %stptr = tt.addptr %1, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    // other value for masked load
    %cnan = arith.constant 0xFF80 : bf16
    %nans = tt.splat %cnan : bf16 -> tensor<128x256xbf16>
    // horizontal mask
    %8 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %9 = arith.cmpi sge, %offset2, %8 : tensor<128xi32>
    %10 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %11 = arith.cmpi slt, %offset2, %10 : tensor<128xi32>
    %12 = arith.andi %9, %11 : tensor<128xi1>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
    %14 = tt.broadcast %13 : tensor<128x1xi1> -> tensor<128x256xi1>
    // vertical mask
    %15 = tt.splat %arg4 : i32 -> tensor<256xi32>
    %16 = arith.cmpi sge, %offset5, %15 : tensor<256xi32>
    %17 = tt.splat %arg5 : i32 -> tensor<256xi32>
    %18 = arith.cmpi slt, %offset5, %17 : tensor<256xi32>
    %19 = arith.andi %16, %18 : tensor<256xi1>
    %20 = tt.expand_dims %19 {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
    %21 = tt.broadcast %20 : tensor<1x256xi1> -> tensor<128x256xi1>
    // combined mask
    %mask = arith.andi %14, %21 : tensor<128x256xi1>
    // offset0 = max(%arg2-2, 0), dim0 = min(%arg3-2, 128) - offset0
    // offset1 = max(%arg4-3, 0), dim1 = min(%arg5-3, 256) - offset1
    // TODO: need reinterpret cast
    %buff = tt.load %ldptr, %mask, %nans : tensor<128x256x!tt.ptr<bf16>>
    tt.store %stptr, %buff, %mask : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel_high_mask(
// CHECK-SAME:          %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, 
// CHECK-SAME:          %[[VAL_0:.*]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}, 
// CHECK-SAME:          %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:.*]]: i32, %[[ARG_10:.*]]: i32, %[[ARG_11:.*]]: i32, %[[ARG_12:.*]]: i32, %[[ARG_13:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} { 
// CHECK-DAG:           %[[VAL_15:.*]] = arith.constant 0xFF80 : bf16
// CHECK-DAG:           %[[VAL_11:.*]] = arith.constant 256 : index
// CHECK-DAG:           %[[VAL_12:.*]] = arith.constant 128 : index
// CHECK-DAG:           %[[VAL_13:.*]] = arith.constant 259 : index
// CHECK-DAG:           %[[VAL_9:.*]] = arith.constant 3 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_14:.*]] = arith.constant 130 : index
// CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 2 : index


// CHECK:           %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [3074], sizes: [128, 256], strides: [1, 1024] : memref<?xbf16> to memref<128x256xbf16, strided<[1, 1024], offset: 3074>>
// CHECK:           %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [3074], sizes: [128, 256], strides: [1, 1024] : memref<?xbf16> to memref<128x256xbf16, strided<[1, 1024], offset: 3074>>
// CHECK:           %[[VAL_18:.*]] = memref.alloc() : memref<128x256xbf16>

// CHECK:           %[[VAL_19:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_20_0:.*]] = arith.maxsi %[[VAL_19]], %[[VAL_10]] : index
// CHECK:           %[[VAL_20:.*]] = arith.minsi %[[VAL_20_0]], %[[VAL_14]] : index
// CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_10]] : index
// CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_14]], %[[VAL_20]] : index

// CHECK:           %[[VAL_23:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_24_0:.*]] = arith.maxsi %[[VAL_23]], %[[VAL_10]] : index
// CHECK:           %[[VAL_24:.*]] = arith.minsi %[[VAL_24_0]], %[[VAL_14]] : index
// CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_24]], %[[VAL_10]] : index

// CHECK:           %[[VAL_26:.*]] = arith.maxsi %[[VAL_21]], %[[VAL_6]] : index
// CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : index
// CHECK:           %[[VAL_28:.*]] = arith.minsi %[[VAL_27]], %[[VAL_25]] : index

// CHECK:           %[[VAL_29:.*]] = arith.index_cast %[[VAL_4]] : i32 to index
// CHECK:           %[[VAL_30_0:.*]] = arith.maxsi %[[VAL_29]], %[[VAL_9]] : index
// CHECK:           %[[VAL_30:.*]] = arith.minsi %[[VAL_30_0]], %[[VAL_13]] : index
// CHECK:           %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_9]] : index
// CHECK:           %[[VAL_32:.*]] = arith.subi %[[VAL_13]], %[[VAL_30]] : index

// CHECK:           %[[VAL_33:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_34_0:.*]] = arith.maxsi %[[VAL_33]], %[[VAL_9]] : index
// CHECK:           %[[VAL_34:.*]] = arith.minsi %[[VAL_34_0]], %[[VAL_13]] : index
// CHECK:           %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_9]] : index

// CHECK:           %[[VAL_36:.*]] = arith.maxsi %[[VAL_31]], %[[VAL_6]] : index
// CHECK:           %[[VAL_37:.*]] = arith.addi %[[VAL_31]], %[[VAL_32]] : index
// CHECK:           %[[VAL_38:.*]] = arith.minsi %[[VAL_37]], %[[VAL_35]] : index

// CHECK:           %[[VAL_39:.*]] = arith.maxsi %[[VAL_26]], %[[VAL_6]] : index
// CHECK:           %[[VAL_40:.*]] = arith.minsi %[[VAL_28]], %[[VAL_12]] : index
// CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_40]], %[[VAL_39]] : index

// CHECK:           %[[VAL_42:.*]] = arith.maxsi %[[VAL_36]], %[[VAL_6]] : index
// CHECK:           %[[VAL_43:.*]] = arith.minsi %[[VAL_38]], %[[VAL_11]] : index
// CHECK:           %[[VAL_44:.*]] = arith.subi %[[VAL_43]], %[[VAL_42]] : index

// CHECK:           %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_41]], %[[VAL_12]] : index
// CHECK:           %[[VAL_46:.*]] = arith.cmpi slt, %[[VAL_44]], %[[VAL_11]] : index
// CHECK:           %[[VAL_47:.*]] = arith.ori %[[VAL_45]], %[[VAL_46]] : i1
// CHECK:           scf.if %[[VAL_47]] {
// CHECK:             linalg.fill ins(%[[VAL_15]] : bf16) outs(%[[VAL_18]] : memref<128x256xbf16>)
// CHECK:           }
// CHECK:           %[[VAL_48:.*]] = memref.subview %[[VAL_16]]{{\[}}%[[VAL_39]], %[[VAL_42]]] {{\[}}%[[VAL_41]], %[[VAL_44]]] [1, 1] : memref<128x256xbf16, strided<[1, 1024], offset: 3074>> to memref<?x?xbf16, strided<[1, 1024], offset: ?>>
// CHECK:           %[[VAL_49:.*]] = memref.subview %[[VAL_18]]{{\[}}%[[VAL_39]], %[[VAL_42]]] {{\[}}%[[VAL_41]], %[[VAL_44]]] [1, 1] : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1], offset: ?>>
// CHECK:           memref.copy %[[VAL_48]], %[[VAL_49]] : memref<?x?xbf16, strided<[1, 1024], offset: ?>> to memref<?x?xbf16, strided<[256, 1], offset: ?>>
// CHECK:           %[[VAL_50:.*]] = bufferization.to_tensor %[[VAL_18]] restrict writable : memref<128x256xbf16>
// CHECK:           %[[VAL_77:.*]] = tensor.extract_slice %[[VAL_50]]{{\[}}%[[VAL_39]], %[[VAL_42]]] {{\[}}%[[VAL_41]], %[[VAL_44]]] [1, 1] : tensor<128x256xbf16> to tensor<?x?xbf16>
// CHECK:           %[[VAL_78:.*]] = memref.subview %[[VAL_17]]{{\[}}%[[VAL_39]], %[[VAL_42]]] {{\[}}%[[VAL_41]], %[[VAL_44]]] [1, 1] : memref<128x256xbf16, strided<[1, 1024], offset: 3074>> to memref<?x?xbf16, strided<[1, 1024], offset: ?>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_77]] in writable %[[VAL_78]] : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[1, 1024], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
