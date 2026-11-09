"""NPU integration tests for Ascend specialization and compile options.

Run this file only with a configured CANN/TorchNPU environment, for example::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -q test_specialization_cache_npu.py

The device-independent unit suite covers SIMT-only and exact cache-key policy;
this file intentionally repeats the three acceptance paths end to end with
real compilation counts and NPU output checks. Current 910B hardware does not
advertise the SIMT execution mode.

Legacy option cases also check argument handling through JIT and Inductor.
"""

import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend with PyTorch

import triton
import triton.language as tl


@triton.jit
def _add_value(src_ptr, dst_ptr, n, value, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    src = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, src + value, mask=mask)


@triton.jit
def _add_annotated_value(src_ptr, dst_ptr, n, value: tl.int32, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    src = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, src + value, mask=mask)


def _clear_kernel_cache(kernel=_add_value):
    device = torch.npu.current_device()
    kernel.device_caches[device][0].clear()


def _run_case(n, src, dst, mode):
    _add_value[(1, )](src, dst, n, 3, BLOCK=32, compile_mode=mode)
    torch.npu.synchronize()
    expected = src[:n] + 3
    torch.testing.assert_close(dst[:n], expected)


def _run_annotated_case(n, value, src, dst, mode):
    _add_annotated_value[(1, )](src, dst, n, value, BLOCK=32, compile_mode=mode)
    torch.npu.synchronize()
    expected = src[:n] + value
    torch.testing.assert_close(dst[:n], expected)


def _assert_aligned_and_unaligned_pair(first, second):
    assert {first.data_ptr() % 16, second.data_ptr() % 16} == {0, first.element_size()}


def _pointer_views(src_base, dst_base, offsets, changed_pointer):
    src_offsets = offsets if changed_pointer in {"src", "both"} else (0, 0)
    dst_offsets = offsets if changed_pointer in {"dst", "both"} else (0, 0)
    src = tuple(src_base[offset:offset + 16] for offset in src_offsets)
    dst = tuple(dst_base[offset:offset + 16] for offset in dst_offsets)

    for name, views in (("src", src), ("dst", dst)):
        if changed_pointer in {name, "both"}:
            _assert_aligned_and_unaligned_pair(*views)
        else:
            assert views[0].data_ptr() == views[1].data_ptr()

    return src, dst


@pytest.fixture(autouse=True)
def _reset_cache_hook():
    _clear_kernel_cache()
    _clear_kernel_cache(_add_annotated_value)
    old_hook = triton.knobs.runtime.jit_cache_hook
    try:
        yield
    finally:
        triton.knobs.runtime.jit_cache_hook = old_hook
        _clear_kernel_cache()
        _clear_kernel_cache(_add_annotated_value)


@pytest.mark.parametrize("mode", ["simd"])
@pytest.mark.parametrize("values", [(16, 17), (17, 16)])
def test_simd_integer_alignment_reuses_compilation(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    _run_case(values[0], src, dst, mode)
    _run_case(values[1], src, dst, mode)

    assert compile_count == 1


@pytest.mark.parametrize("mode", ["simd"])
@pytest.mark.parametrize("offsets", [(0, 1), (1, 0)], ids=["aligned-first", "unaligned-first"])
@pytest.mark.parametrize("changed_pointer", ["src", "dst", "both"], ids=["src-only", "dst-only", "src-and-dst"])
def test_simd_pointer_alignment_reuses_compilation(mode, offsets, changed_pointer):
    src_base = torch.arange(17, dtype=torch.int32, device="npu")
    dst_base = torch.empty_like(src_base)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    src, dst = _pointer_views(src_base, dst_base, offsets, changed_pointer)
    _run_case(16, src[0], dst[0], mode)
    _run_case(16, src[1], dst[1], mode)

    assert compile_count == 1


@pytest.mark.parametrize("mode", ["simd"])
@pytest.mark.parametrize("values", [(1, 2), (2, 1)])
def test_simd_integer_one_still_recompiles(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    _run_case(values[0], src, dst, mode)
    _run_case(values[1], src, dst, mode)

    assert compile_count == 2


@pytest.mark.parametrize("mode", ["simd"])
@pytest.mark.parametrize("values", [(1, 2), (2, 1)])
def test_simd_annotated_integer_one_is_constexpr_and_recompiles(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_records = []

    def record_compile(**kwargs):
        compile_info = kwargs["compile"]
        compile_records.append((compile_info["signature"], compile_info["constants"]))

    triton.knobs.runtime.jit_cache_hook = record_compile
    _run_annotated_case(32, values[0], src, dst, mode)
    _run_annotated_case(32, values[1], src, dst, mode)

    assert len(compile_records) == 2
    for value, (signature, constants) in zip(values, compile_records):
        if value == 1:
            assert signature["value"] == "constexpr"
            assert constants[(3, )] == 1
        else:
            assert signature["value"] == "i32"
            assert (3, ) not in constants


@pytest.mark.parametrize("compile_graph", [False, True], ids=["jit", "inductor"])
@pytest.mark.parametrize(
    "name, value",
    [("use_bytecode", False), ("warp_size", 1), ("stream", 0), ("enable_vf_fusion", True), ("enable_vf_fusion", False)],
)
def test_legacy_option_is_not_a_kernel_argument(compile_graph, name, value):
    if compile_graph:
        from torch.utils._triton import has_triton_package

        # Older PyTorch checks triton_key, which Triton 3.6 no longer exposes.
        if not has_triton_package():
            pytest.skip(f"PyTorch {torch.__version__} cannot recognize Triton {triton.__version__} for Inductor")
        # Register NPU support before Dynamo checks whether Triton is available.
        import torch_npu._inductor  # noqa: F401

        from torch._inductor.codecache import CacheBase

        # Some TorchNPU builds still import triton_key when constructing cache keys.
        try:
            CacheBase.get_system()
        except ImportError as error:
            if error.name != "triton.compiler.compiler" or "triton_key" not in str(error):
                raise
            pytest.skip(f"TorchNPU {torch_npu.__version__} requires the legacy Triton triton_key API")

    legacy_options = {name: value}

    def add_one(x):
        result = torch.empty_like(x)
        _add_value[(1, )](x, result, x.numel(), 1, BLOCK=32, **legacy_options)
        return result

    run = torch.compile(add_one, fullgraph=True) if compile_graph else add_one
    x = torch.arange(17, device="npu", dtype=torch.float32)

    torch.testing.assert_close(run(x), x + 1)
