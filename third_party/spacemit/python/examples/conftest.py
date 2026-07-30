import pytest
import triton
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


def empty_decorator(func):
    return func


pytest.mark.interpreter = empty_decorator


@pytest.fixture
def device(request):
    return "cpu"


# this fixture is used for test_trans_4d && test_trans_reshape
@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


tests_supported = {
    "test_store_eviction_policy",
    "test_unary_op",
    "test_umulhi",
    "test_for_iv",
    "test_trans_2d",
    "test_math_op",
    "test_math_fma_op",
    "test_abs",
    "test_call",
    "test_vectorization",
    "test_convert_float16_to_float32",
    "test_index1d",
    "test_shift_op",
    "test_full",
    "test_floordiv",
    "test_empty_kernel",
    "test_if_return",
    "test_value_specialization",
    "test_propagate_nan",
    "test_clamp",
    "test_clamp_symmetric",
    "test_store_cache_modifier",
    "test_permute",
    "test_broadcast",
    "test_precise_math",
    "test_vectorization_hints",
    "test_dot",
    "test_value_specialization_overflow",
    "test_bitwise_op",
    "test_const",
    "test_unary_math",
    "test_dot_mulbroadcasted",
    "test_masked_load_scalar",
    "test_enable_fp_fusion",
    "test_load_cache_modifier",
    "test_dot_without_load",
    "test_cat",
    "test_addptr",
    "test_transpose",
    "test_trans_4d",
    "test_unsplat",
    "test_arange",
}


def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason="CPU backend does not support it yet")
    # There is a dependency issue on build machine which breaks bfloat16
    skip_marker_bfloat = pytest.mark.skip(reason="bfloat16 linking issue")
    skip_marker_tf32 = pytest.mark.skip(reason="tf32 is not supported on CPU")
    skip_marker_float8 = pytest.mark.skip(reason="float8 is not supported on CPU")

    for item in items:
        test_func_name = item.originalname if item.originalname else item.name

        test_file = str(item.fspath)
        if test_file.endswith("test_core.py") and test_func_name not in tests_supported:
            item.add_marker(skip_marker)
            continue

        if "parametrize" in item.keywords:
            for param_name, param_value in item.callspec.params.items():
                if (param_name.startswith('dtype') or param_name.endswith('dtype')) and param_value == 'bfloat16':
                    item.add_marker(skip_marker_bfloat)
                if param_name.startswith('input_precision') and param_value.startswith('tf32'):
                    item.add_marker(skip_marker_tf32)
                if (param_name.startswith('dtype') or param_name.endswith('dtype')) and ('float8' in str(param_value)):
                    item.add_marker(skip_marker_float8)
