import pytest
import os
import tempfile
import triton
from triton.backends.ztc.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


def empty_decorator(func):
    return func


pytest.mark.interpreter = empty_decorator


@pytest.fixture
def device(request):
    return "cpu"


tests_not_supported = {
    "test_bin_op", "test_split_to_scalar", "test_interleave_scalars", "test_pointer_arguments",
    "test_masked_load_shared_memory", "test_bin_op_constexpr", "test_num_warps_pow2", "test_math_divide_op",
    "test_atomic_rmw_predicate", "test_tensor_atomic_rmw_block", "test_nested_if_else_return", "test_ptx_cast",
    "test_compare_op", "test_maxnreg", "test_join_scalars", "test_join_with_mma", "test_interleave", "test_slice",
    "test_where", "test_math_erf_op", "test_abs_fp8", "test_shapes_as_params", "test_transpose", "test_where_broadcast",
    "test_noinline", "test_cast", "test_store_constant", "test_reduce", "test_reduce1d",
    "test_optimize_thread_locality", "test_scan_layouts", "test_reduce_layouts", "test_store_op", "test_convert1d",
    "test_chain_reduce", "test_generic_reduction", "test_trans_4d", "test_dot3d", "test_constexpr", "test_arange",
    "test_masked_load", "test_reshape", "test_trans_reshape", "test_if", "test_if_call", "test_convert2d",
    "test_convertmma2mma", "test_dot_max_num_imprecise_acc", "test_propagate_nan", "test_clamp_symmetric",
    "test_temp_var_in_loop", "test_math_extern", "test_cdiv"  # Failed due to precision issues in integer division.
}

# probably different version of MLIR on the nightly build machine is complaining
# about unregistered dialect for llvm.intr.assume, pre-commit checks are passing
tests_not_supported.add("test_assume")

# List of files that pytest should ignore
# TODO: FIX THESE TESTS
collect_ignore = [
    'test_core.py',
    'test_early_return.py',
    'test_load_2d_tensor_block.py',
    'test_mask.py',
    'test_modulo.py',
    'test_nested_loops.py',
    'test_reduce.py',  # Only test gpu compile. No need fix
    'test_scalar_store.py',
    'test_sign_extend.py',
    'test_softmax.py',
    'test_tensor_index_iterargs.py',
    "test_dot_scaled.py",
    "test_flip.py",
    "test_sort.py",
]


def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason="CPU backend does not support it yet")
    # There is a dependency issue on build machine which breaks bfloat16
    skip_marker_bfloat = pytest.mark.skip(reason="bfloat16 linking issue")
    skip_marker_tf32 = pytest.mark.skip(reason="tf32 is not supported on CPU")
    skip_marker_float8 = pytest.mark.skip(reason="float8 is not supported on CPU")

    for item in items:
        test_func_name = item.originalname if item.originalname else item.name

        if test_func_name in tests_not_supported:
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
