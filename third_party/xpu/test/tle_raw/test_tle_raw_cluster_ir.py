"""IR-level tests for `triton_xpu.raw` (tle.raw on the xpu3 cluster path).

These only exercise the compiler, so they run without XPU hardware:
  - `tle.raw.call` must emit a `triton_xpu.raw` at the ttir level
  - `convert-tritonxpu-to-llvm` must declare the payload the op names and
    replace the raw op with an `always_inline` `llvm.call`

The payload body is deliberately NOT in the module: it is LLVM 19 IR text that
the backend compiles for its own arch and merges in at the LLVM 19 stage (see
`triton/experimental/tle/raw/merge.py`, and test_tle_raw_deferred.py for that
half). All this pass produces is the declaration the call binds to, so the merge
can check the payload against it.

The SDNN counterpart lives only in the internal tree (not shipped in the public tree); the two paths never meet in one
kernel (SDNN goes through the is_sdnn branch of make_llir, cluster through the
TritonXPU one).

Note on reading IR back: FlagTree's XPU build defines TRITON_CONCEAL_IR, which
makes `str(module)` return an empty string. `_ir_text` falls back to
`create_location_snapshot`, which writes the module to a file and is not gated.
"""

import pytest

from triton._C.libtriton import gluon_ir, ir, xpu

XPU_ARCH = 3
BUFFER_LEN = 512

pytestmark = pytest.mark.no_xpu_required


@pytest.fixture(scope="module")
def ctx():
    context = ir.context()
    ir.load_dialects(context)
    xpu.load_dialects(context)
    return context


def _ir_text(mod, tmp_path, name="dump.mlir"):
    """Module as text, working with or without TRITON_CONCEAL_IR."""
    text = str(mod)
    if text:
        return text
    path = tmp_path / name
    mod.create_location_snapshot(str(path))
    return path.read_text()


def _parse(ctx, tmp_path, src):
    path = tmp_path / "case.mlir"
    path.write_text(src)
    mod = ir.parse_mlir_module(str(path), ctx)
    mod.context = ctx
    return mod


def _lower(ctx, tmp_path, src):
    mod = _parse(ctx, tmp_path, src)
    pm = ir.pass_manager(ctx)
    xpu.passes.ttxpuir.add_convert_tritonxpu_to_llvm_pass(pm, XPU_ARCH, BUFFER_LEN, False)
    pm.run(mod, "to-llvm")
    return _ir_text(mod, tmp_path, "lowered.mlir")


def test_builder_emits_raw_op_at_ttir_level(ctx, tmp_path):
    """`tle.raw.call` without `is_sdnn` goes through GluonOpBuilder.create_xpu_raw."""
    builder = gluon_ir.GluonOpBuilder(ctx)
    mod = builder.create_module()
    ptr_ty = builder.get_ptr_ty(builder.get_float_ty(), 1)
    i32_ty = builder.get_int32_ty()
    fn = builder.get_or_insert_function(mod, "kernel", builder.get_function_ty([ptr_ty, ptr_ty, i32_ty], []), "public",
                                        False)
    mod.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    builder.create_xpu_raw("my_scale", "deadbeef", [fn.args(0), fn.args(1), fn.args(2)])
    builder.ret([])

    out = _ir_text(mod, tmp_path)
    assert 'triton_xpu.raw "my_scale"(%arg0, %arg1, %arg2)' in out
    assert 'source_id = "deadbeef"' in out


def test_lowering_declares_the_payload_and_calls_it(ctx, tmp_path):
    # `tt.ptr<f32>` converts to `!llvm.ptr<1>`, which is also what xpu-clang emits
    # for a `_global_ptr_` parameter -- the payload has to agree or the merge
    # rejects it.
    src = '''
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 1 : i32} {
  tt.func public @raw_kernel(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %n: i32) {
    triton_xpu.raw "my_scale"(%out, %in, %n)
      {source_id = "deadbeef"} : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
    tt.return
  }
}
'''
    out = _lower(ctx, tmp_path, src)

    # External declaration, no payload body: the backend merges the body in at
    # the LLVM 19 stage and checks it against this signature.
    assert "llvm.func @my_scale" in out
    assert "internal @my_scale" not in out
    assert "always_inline" in out
    assert "llvm.call @my_scale" in out
    assert 'triton_xpu.raw "' not in out
    # The operands are passed through as they come.
    assert "(!llvm.ptr<1>, !llvm.ptr<1>, i32) -> ()" in out


CXX_PAYLOAD = '''
#include "xpu/kernel/xtdk.h"
extern "C" __device__ void my_scale(_global_ptr_ float* out,
                                    _global_ptr_ const float* in, int n) {
  for (int i = 0; i < n; ++i) out[i] = in[i] * 2.0f;
}
'''


def test_cxx_payload_lowers_to_a_declared_call(ctx, tmp_path):
    """The C++ path: .xpu source -> LLVM IR -> a call the backend binds later."""
    import triton.experimental.tle as tle
    from triton.experimental.tle.raw.merge import record_raw_payloads
    from triton.experimental.tle.raw.source_store import clear_pending_sources

    @tle.raw.dialect("xpu3", source=CXX_PAYLOAD, arch=XPU_ARCH)
    def my_scale(out, inp, n):
        ...

    clear_pending_sources()
    try:
        try:
            source_id = my_scale.register_payload()
        except RuntimeError as e:
            pytest.skip(f"no usable XPU clang for the C++ payload path: {e}")

        src = f'''
module attributes {{"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 1 : i32}} {{
  tt.func public @raw_kernel(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %n: i32) {{
    triton_xpu.raw "my_scale"(%out, %in, %n)
      {{source_id = "{source_id}"}} : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
    tt.return
  }}
}}
'''
        mod = _parse(ctx, tmp_path, src)
        metadata = {}
        record_raw_payloads(mod, metadata)
        assert metadata["tle_raw_payloads"] == [("my_scale", source_id)]

        pm = ir.pass_manager(ctx)
        xpu.passes.ttxpuir.add_convert_tritonxpu_to_llvm_pass(pm, XPU_ARCH, BUFFER_LEN, False)
        pm.run(mod, "to-llvm")

        out = _ir_text(mod, tmp_path)
        assert "llvm.call @my_scale" in out
        assert "always_inline" in out
        # The payload's body is merged in later, not carried through MLIR.
        assert "llvm.fmul" not in out
    finally:
        clear_pending_sources()
