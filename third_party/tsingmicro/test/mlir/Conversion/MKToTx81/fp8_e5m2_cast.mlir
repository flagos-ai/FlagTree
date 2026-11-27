//===----------------------------------------------------------------------===//
// Test File for MK to Tx81 Conversion: f32 <-> f8E5M2 bit manipulation
//
// TX81 has no native f32 <-> f8E5M2 conversion instruction, so MKToTx81
// expands `arith.truncf f32 -> f8E5M2` and `arith.extf f8E5M2 -> f32` into
// integer bit manipulation (RNE, Inf/NaN preserved for E5M2).
//===----------------------------------------------------------------------===//

// RUN: tsingmicro-opt --expand-strided-metadata --lower-affine --mk-to-tx81 --cse --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

func.func @trunc_f32_to_e5m2() {
  %src = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<64xf8E5M2>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%src : memref<64xf32>) outs(%dst : memref<64xf8E5M2>) {
  ^bb0(%in: f32, %out: f8E5M2):
    %0 = arith.truncf %in : f32 to f8E5M2
    linalg.yield %0 : f8E5M2
  }
  return
}

// The truncf must be fully replaced by integer ops. 0x7C is the E5M2 Inf
// encoding and 0x7F the canonical NaN, so both special-case constants must
// be materialized (E4M3FN instead folds Inf into NaN).
// CHECK-LABEL: func.func @trunc_f32_to_e5m2
// CHECK-NOT:   arith.truncf
// CHECK-DAG:   arith.constant 124 : i32
// CHECK-DAG:   arith.constant 127 : i32
// CHECK-DAG:   arith.constant 112 : i32
// CHECK-DAG:   arith.bitcast %{{.*}} : f32 to i32
// CHECK-DAG:   arith.bitcast %{{.*}} : i8 to f8E5M2
// CHECK:       memref.store {{.*}} : memref<64xf8E5M2>
// CHECK-NOT:   arith.truncf

// -----

#map = affine_map<(d0) -> (d0)>

func.func @ext_e5m2_to_f32() {
  %src = memref.alloc() {alignment = 64 : i64} : memref<64xf8E5M2>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%src : memref<64xf8E5M2>) outs(%dst : memref<64xf32>) {
  ^bb0(%in: f8E5M2, %out: f32):
    %0 = arith.extf %in : f8E5M2 to f32
    linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: func.func @ext_e5m2_to_f32
// CHECK-NOT:   arith.extf
// CHECK-DAG:   arith.bitcast %{{.*}} : f8E5M2 to i8
// CHECK-DAG:   arith.extui %{{.*}} : i8 to i32
// CHECK-DAG:   arith.constant 113 : i32
// CHECK-DAG:   arith.constant 2139095040 : i32
// CHECK-DAG:   arith.constant 2143289344 : i32
// CHECK:       memref.store {{.*}} : memref<64xf32>
// CHECK-NOT:   arith.extf

// -----

#map = affine_map<(d0) -> (d0)>

func.func @roundtrip_e5m2() {
  %src = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%src : memref<64xf32>) outs(%dst : memref<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.truncf %in : f32 to f8E5M2
    %1 = arith.extf %0 : f8E5M2 to f32
    linalg.yield %1 : f32
  }
  return
}

// Both casts must be expanded in the same lowering: the generic is erased
// after the loop nest is generated, so lowering them one kind at a time would
// leave the other cast in the scf.for body, where it cannot be lowered
// (float8 is represented as i8 in LLVM).
// CHECK-LABEL: func.func @roundtrip_e5m2
// CHECK-NOT:   arith.truncf
// CHECK-NOT:   arith.extf
// CHECK:       memref.store {{.*}} : memref<64xf32>
// CHECK-NOT:   arith.truncf
// CHECK-NOT:   arith.extf

// -----

#map = affine_map<(d0) -> (d0)>

func.func @mixed_e4m3fn_e5m2() {
  %fa = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  %be = memref.alloc() {alignment = 64 : i64} : memref<64xf8E5M2>
  %outa = memref.alloc() {alignment = 64 : i64} : memref<64xf8E4M3FN>
  %outb = memref.alloc() {alignment = 64 : i64} : memref<64xf32>
  linalg.generic {indexing_maps = [#map, #map, #map, #map],
                  iterator_types = ["parallel"]}
      ins(%fa, %be : memref<64xf32>, memref<64xf8E5M2>)
      outs(%outa, %outb : memref<64xf8E4M3FN>, memref<64xf32>) {
  ^bb0(%in0: f32, %in1: f8E5M2, %out0: f8E4M3FN, %out1: f32):
    %0 = arith.truncf %in0 : f32 to f8E4M3FN
    %1 = arith.extf %in1 : f8E5M2 to f32
    linalg.yield %0, %1 : f8E4M3FN, f32
  }
  return
}

// Casts of different float8 formats in the same body must both be expanded,
// not just the first one matched.
// CHECK-LABEL: func.func @mixed_e4m3fn_e5m2
// CHECK-NOT:   arith.truncf
// CHECK-NOT:   arith.extf
// CHECK:       memref.store
// CHECK-NOT:   arith.truncf
// CHECK-NOT:   arith.extf
