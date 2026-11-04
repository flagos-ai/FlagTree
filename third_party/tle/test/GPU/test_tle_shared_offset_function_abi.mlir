// RUN: triton-opt %s -triton-tle-shared-offset-function-abi -canonicalize | FileCheck %s
// RUN: triton-opt %s -triton-tle-shared-offset-function-abi -triton-tle-shared-offset-function-abi -canonicalize | FileCheck %s
// Shared pointer formals, including those nested in memdesc structures, cross
// a noinline boundary as arena-relative offsets. Global pointers stay pointers.
module {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  // CHECK-LABEL: llvm.func internal @read(%{{.*}}: i32, %{{.*}}: !llvm.ptr<1>)
  // CHECK: llvm.mlir.addressof @global_smem
  // CHECK: llvm.add
  // CHECK: llvm.inttoptr
  // CHECK: llvm.load
  llvm.func internal @read(%shared: !llvm.ptr<3> {llvm.align = 16 : i64, llvm.nonnull}, %global: !llvm.ptr<1>) -> i32 attributes {passthrough = ["noinline"]} {
    %value = llvm.load %shared : !llvm.ptr<3> -> i32
    llvm.store %value, %global : i32, !llvm.ptr<1>
    llvm.return %value : i32
  }
  // CHECK-LABEL: llvm.func internal @nested(%{{.*}}: !llvm.struct<(i32, struct<(i32, ptr<1>)>)>)
  // CHECK: llvm.call @read({{.*}}) : (i32, !llvm.ptr<1>) -> i32
  llvm.func internal @nested(%view: !llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>) -> i32 attributes {passthrough = ["noinline"]} {
    %shared = llvm.extractvalue %view[0] : !llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>
    %global = llvm.extractvalue %view[1, 1] : !llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>
    %value = llvm.call @read(%shared, %global) : (!llvm.ptr<3>, !llvm.ptr<1>) -> i32
    llvm.return %value : i32
  }
  // CHECK-LABEL: llvm.func @external_api(!llvm.ptr<3>)
  llvm.func @external_api(!llvm.ptr<3>)
  // CHECK-LABEL: llvm.func @entry
  // CHECK: llvm.call @nested({{.*}}) : (!llvm.struct<(i32, struct<(i32, ptr<1>)>)>) -> i32
  // CHECK: llvm.call @external_api({{.*}}) : (!llvm.ptr<3>) -> ()
  llvm.func @entry(%global: !llvm.ptr<1>) attributes {nvvm.kernel} {
    %base = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
    %shared = llvm.getelementptr %base[512] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
    %zero = llvm.mlir.constant(0 : i32) : i32
    %undef = llvm.mlir.undef : !llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>
    %one = llvm.insertvalue %shared, %undef[0] : !llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>
    %two = llvm.insertvalue %zero, %one[1, 0] : !llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>
    %view = llvm.insertvalue %global, %two[1, 1] : !llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>
    %value = llvm.call @nested(%view) : (!llvm.struct<(ptr<3>, struct<(i32, ptr<1>)>)>) -> i32
    llvm.call @external_api(%shared) : (!llvm.ptr<3>) -> ()
    llvm.return
  }
}
