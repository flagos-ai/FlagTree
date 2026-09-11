"""IR-level tests for `triton_xpu.raw` (tle.raw on the xpu3 cluster path).

These only exercise the compiler, so they run without XPU hardware:
  - `tle.raw.call` must emit a `triton_xpu.raw` at the ttir level
  - `convert-tritonxpu-to-llvm` must splice the payload into the module and
    replace the raw op with an `always_inline` `llvm.call`
  - `tritonxpu-materialize-deferred-raw` must fill a deferred payload in first

The SDNN counterpart (`sdnn.raw`, `is_sdnn=True`) has no frontend in FlagTree --
its builder needs TritonSDNN's IR headers, which this tree does not ship -- so
only the cluster path is covered here.

Note on reading IR back: FlagTree's XPU build defines TRITON_CONCEAL_IR, which
makes `str(module)` return an empty string. `_ir_text` falls back to
`create_location_snapshot`, which writes the module to a file and is not gated.
"""

import json

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


# `tt.ptr<f32>` converts to `!llvm.ptr<1>`, which is also what xpu-clang emits
# for a `_global_ptr_` parameter -- the payload has to agree or the call is
# rejected.
MLIR_PAYLOAD = ("llvm.func @my_scale(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, "
                "%arg2: i32) { llvm.return }")

LL_PAYLOAD = ("define void @my_scale(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %n) {\\0A"
              "  ret void\\0A"
              "}\\0A")


def _kernel_module(payload, extra_attrs=""):
    return f'''
module attributes {{"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 1 : i32}} {{
  tt.func public @raw_kernel(%out: !tt.ptr<f32>, %in: !tt.ptr<f32>, %n: i32) {{
    triton_xpu.raw "my_scale"(%out, %in, %n)
      {{llvm_ir = "{payload}"{extra_attrs}}} : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
    tt.return
  }}
}}
'''


def _lower(ctx, tmp_path, src, sources=None):
    mod = _parse(ctx, tmp_path, src)
    pm = ir.pass_manager(ctx)
    if sources is not None:
        xpu.passes.ttxpuir.add_tritonxpu_materialize_deferred_raw_pass(pm, sources)
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
    builder.create_xpu_raw("my_scale", MLIR_PAYLOAD, [fn.args(0), fn.args(1), fn.args(2)])
    builder.ret([])

    out = _ir_text(mod, tmp_path)
    assert 'triton_xpu.raw "my_scale"(%arg0, %arg1, %arg2)' in out
    assert "llvm_ir" in out


def test_builder_emits_a_deferred_raw_op(ctx, tmp_path):
    builder = gluon_ir.GluonOpBuilder(ctx)
    mod = builder.create_module()
    ptr_ty = builder.get_ptr_ty(builder.get_float_ty(), 1)
    fn = builder.get_or_insert_function(mod, "kernel", builder.get_function_ty([ptr_ty], []), "public", False)
    mod.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    builder.create_xpu_raw_deferred("my_scale", "deadbeef", [fn.args(0)])
    builder.ret([])

    out = _ir_text(mod, tmp_path)
    assert 'triton_xpu.raw "my_scale"(%arg0)' in out
    assert 'triton_xpu.raw_source_id = "deadbeef"' in out
    assert 'llvm_ir = ""' in out


@pytest.mark.parametrize("payload", [MLIR_PAYLOAD, LL_PAYLOAD], ids=["mlir", "llvmir"])
def test_lowering_emits_always_inline_call(ctx, tmp_path, payload):
    out = _lower(ctx, tmp_path, _kernel_module(payload))

    # Payload spliced in as an internal definition, so the backend cannot mistake
    # it for the kernel entry point when it looks for the external one.
    assert "llvm.func internal @my_scale" in out
    assert "triton_xpu.raw_payload" in out
    assert "always_inline" in out
    assert "llvm.call @my_scale" in out
    assert 'triton_xpu.raw "' not in out


def test_deferred_then_lowering_emits_the_call(ctx, tmp_path):
    src = _kernel_module("", extra_attrs=', triton_xpu.raw_source_id = "deadbeef"')
    out = _lower(ctx, tmp_path, src, sources={"deadbeef": MLIR_PAYLOAD})

    assert "raw_source_id" not in out
    assert "llvm.call @my_scale" in out
    assert "always_inline" in out


def test_materialize_pass_reports_an_unknown_source_id(ctx, tmp_path):
    src = _kernel_module("", extra_attrs=', triton_xpu.raw_source_id = "deadbeef"')
    with pytest.raises(RuntimeError):
        _lower(ctx, tmp_path, src, sources={"other": MLIR_PAYLOAD})


def test_lowering_without_materialization_is_reported(ctx, tmp_path):
    src = _kernel_module("", extra_attrs=', triton_xpu.raw_source_id = "deadbeef"')
    with pytest.raises(RuntimeError):
        _lower(ctx, tmp_path, src)


CXX_PAYLOAD = '''
#include "xpu/kernel/xtdk.h"
extern "C" __device__ void my_scale(_global_ptr_ float* out,
                                    _global_ptr_ const float* in, int n) {
  for (int i = 0; i < n; ++i) out[i] = in[i] * 2.0f;
}
'''


def _cxx_payload():
    """Compile a cluster C++ payload with the real clang, or skip if unavailable."""
    import triton.experimental.tle as tle

    @tle.raw.dialect("xpu3", source=CXX_PAYLOAD, arch=XPU_ARCH)
    def my_scale(out, inp, n):
        ...

    try:
        return my_scale.make_llvm()
    except RuntimeError as e:
        pytest.skip(f"no usable XPU clang for the C++ payload path: {e}")


def test_cxx_payload_compiles_and_lowers(ctx, tmp_path):
    """The whole C++ path: .xpu source -> LLVM IR -> imported -> llvm.call."""
    out = _lower(ctx, tmp_path, _kernel_module(json.dumps(_cxx_payload())[1:-1]))

    # The payload body came along, not just its declaration.
    assert "llvm.fmul" in out
    assert "llvm.call @my_scale" in out
    assert "always_inline" in out
