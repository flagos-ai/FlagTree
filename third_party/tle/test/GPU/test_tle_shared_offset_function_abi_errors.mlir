// RUN: triton-opt %s -split-input-file -triton-tle-shared-offset-function-abi -verify-diagnostics
module {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // expected-error @+1 {{Shared offset ABI requires direct calls only}}
  llvm.func internal @taken(%shared: !llvm.ptr<3>) {
    llvm.return
  }
  llvm.func @entry() -> !llvm.ptr {
    %address = llvm.mlir.addressof @taken : !llvm.ptr
    llvm.return %address : !llvm.ptr
  }
}
// -----
module {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // expected-error @+1 {{Shared offset ABI requires non-variadic functions}}
  llvm.func internal @variadic(%shared: !llvm.ptr<3>, ...) {
    llvm.return
  }
}
// -----
module {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32} : !llvm.array<0 x i8>
  // expected-error @+1 {{Shared offset ABI cannot change copy/return argument conventions}}
  llvm.func internal @copy(%shared: !llvm.ptr<3> {llvm.byval = i32}) {
    llvm.return
  }
}
