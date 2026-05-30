// RUN: triton-opt %s --triton-to-hivm | FileCheck %s

// CHECK-LABEL: tt.func @triton_func
tt.func @triton_func() {
    ascend.custom "sync_block_set" {str_args = ["vector", 1 : i32]}
    ascend.custom "sync_block_wait" {str_args = ["vector", 1 : i32]}
    ascend.custom "sync_block_set" {str_args = ["cube", 2 : i32]}
    ascend.custom "sync_block_wait" {str_args = ["cube", 2 : i32]}
    ascend.custom "sync_block_all" {str_args = ["all_cube", 1 : i32]}
    ascend.custom "sync_block_all" {str_args = ["all_vector", 1 : i32]}
    ascend.custom "sync_block_all" {str_args = ["all", 1 : i32]}
    tt.return
}
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE2>] flag = 1
// CHECK: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_MTE2>] flag = 2
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_MTE2>] flag = 2
// CHECK: hivm.hir.sync_block[<ALL_CUBE>, 1 : i16] tcube_pipe = <PIPE_FIX>
// CHECK: hivm.hir.sync_block[<ALL_VECTOR>, 1 : i16] tvector_pipe = <PIPE_MTE3>
// CHECK: hivm.hir.sync_block[<ALL>, 1 : i16] tcube_pipe = <PIPE_FIX> tvector_pipe = <PIPE_MTE3>
