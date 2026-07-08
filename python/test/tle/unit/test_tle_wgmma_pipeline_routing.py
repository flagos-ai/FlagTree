import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, make_backend

_WGMMA_PIPELINE_MODE_ATTR = "tle.wgmma_pipeline_mode"
_USER_PROMISE_MODE = "user_promise"
_USER_PROMISE_MODE_ATTRS = (
    f'"{_WGMMA_PIPELINE_MODE_ATTR}" = "{_USER_PROMISE_MODE}"',
    f'{_WGMMA_PIPELINE_MODE_ATTR} = "{_USER_PROMISE_MODE}"',
)
_HOPPER_TARGET = GPUTarget("cuda", 90, 32)


def _require_cuda():
    try:
        import torch

        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "cuda":
            pytest.skip(f"CUDA Hopper backend is required, got {target.backend}")
        if int(target.arch) < 90:
            pytest.skip(f"CUDA Hopper backend is required, got sm{target.arch}")
        torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_cuda()


@triton.jit
def _no_trigger_kernel(out):
    tl.store(out, 0)


@triton.jit
def _alloc_barriers_trigger_kernel():
    tle.gpu.alloc_barriers(2)


@triton.jit
def _alloc_barrier_trigger_kernel():
    tle.gpu.alloc_barrier()


@triton.jit
def _wgmma_wait_trigger_kernel():
    acc = tl.zeros((16, 16), tl.float32)
    tle.gpu.wgmma_wait(0, acc)


@triton.jit
def _ws_call_default(out):
    tl.store(out, 1)


@triton.jit
def _ws_call_worker(out):
    tl.store(out, 2)


@triton.jit
def _warp_specialize_call_lower_kernel(out):
    tle.gpu.warp_specialize(
        [
            (_ws_call_default, (out, )),
            (_ws_call_worker, (out, )),
        ],
        [4],
        [168],
    )


@triton.jit
def _ws_barrier_default(bar):
    tle.gpu.barrier_arrive(bar, phaseIdx=0)


@triton.jit
def _ws_barrier_worker(bar, out):
    tle.gpu.barrier_wait(bar, phaseIdx=0)
    tl.store(out, 1)


@triton.jit
def _warp_specialize_barrier_inline_kernel(out):
    bar = tle.gpu.alloc_barrier()
    tle.gpu.warp_specialize(
        [
            (_ws_barrier_default, (bar, )),
            (_ws_barrier_worker, (bar, out)),
        ],
        [4],
        [168],
    )


def _make_ttir(kernel, signature=None):
    backend = make_backend(_HOPPER_TARGET)
    options = backend.parse_options({"num_warps": 4})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    src = ASTSource(fn=kernel, signature=signature or {}, constexprs={})
    module = src.make_ir(
        _HOPPER_TARGET,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )
    return str(module)


def _has_user_promise_mode_attr(ttir):
    return any(attr in ttir for attr in _USER_PROMISE_MODE_ATTRS)


@pytest.mark.parametrize(
    ("kernel", "signature", "routes_user_promise"),
    [
        (_no_trigger_kernel, {"out": "*i32"}, False),
        (_alloc_barriers_trigger_kernel, {}, True),
        (_alloc_barrier_trigger_kernel, {}, True),
        (_wgmma_wait_trigger_kernel, {}, True),
    ],
)
def test_tle_wgmma_pipeline_route_marker_exact_api_list(kernel, signature, routes_user_promise):
    ttir = _make_ttir(kernel, signature)

    assert _has_user_promise_mode_attr(ttir) is routes_user_promise


def test_tle_warp_specialize_keeps_call_lowering_without_user_promise_marker():
    ttir = _make_ttir(_warp_specialize_call_lower_kernel, {"out": "*i32"})

    assert "ttg.warp_specialize" in ttir
    assert "tt.call" in ttir
    assert not _has_user_promise_mode_attr(ttir)


def test_tle_warp_specialize_inlines_with_user_promise_marker():
    ttir = _make_ttir(_warp_specialize_barrier_inline_kernel, {"out": "*i32"})

    assert "ttg.warp_specialize" in ttir
    assert _has_user_promise_mode_attr(ttir)
    assert "tt.call" not in ttir
