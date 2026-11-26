import importlib.util
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))


def _load_driver_module():
    driver_path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_driver_under_test", driver_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()


def _mock_backend_func(name, *args):
    return f"/* {name}: {args} */"


def _make_metadata():
    return SimpleNamespace(
        target=driver.GPUTarget("npu", "Ascend910B3", 0),
        workspace_size=0,
        lock_init_value=0,
        sync_block_lock_layout=0,
        bs_task_type=0,
        mix_mode="aiv",
        shared=0,
        compile_on_910_95=False,
        parallel_mode="",
        is_pure_simt=False,
        debug=False,
        coalesce_factor=1,
        coalesce_axis=-1,
        coalesce_grid_ceil_div=False,
        has_auto_blockify_blacklist_op=False,
        row_coalescing_applied=False,
        program_grid_transforms=None,
        program_grid_mapping_applied=False,
        auto_blockify_enabled=False,
        ptsm_cap_authorized=False,
    )


def _make_program_grid_contract(*transforms):
    return {
        "version": 2,
        "extent_source": "runtime_original_grid",
        "hidden_extent_axes": [0, 1],
        "hidden_argument_order": ["originalGridX", "originalGridY"],
        "hidden_argument_types": ["i32", "i32"],
        "transforms": list(transforms),
    }


def _split_launch_functions(src):
    c_abi_part, cpp_part = src.split("static void _launch(", maxsplit=1)
    return c_abi_part, "static void _launch(" + cpp_part


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_exposes_triton_launch_kernel(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32", 2: "i32"},
        metadata=_make_metadata(),
    )

    assert 'void triton_launch_kernel(' in src
    assert 'const void* const* kernel_args, const size_t* arg_sizes, int num_args' in src
    assert 'std::vector<std::vector<char>> copied_kernel_args;' in src
    assert 'std::vector<size_t> launch_arg_sizes;' in src
    assert 'std::vector<char> launch_args(total_size, 0);' in src
    assert 'memcpy(launch_args.data() + grid_offset, &gridX, sizeof(int32_t));' in src
    _mock_ffts.assert_called_once_with("Ascend910B3")


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_resolves_npu_utils_from_active_cache_root(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    cache_key = "NPU_UTILS_CACHE_KEY"

    def make_utils(cache_root):
        return SimpleNamespace(
            get_aivector_core_num=lambda: 40,
            get_aicore_num=lambda: 20,
            get_so_path=lambda: f"{cache_root}/{cache_key}/npu_utils.so",
        )

    producer_utils = make_utils("/producer/cache")
    consumer_utils = make_utils("/consumer/cache")
    # make_launcher currently reads NPUUtils once for core counts and once for
    # the shared-object path.
    mock_npu_utils.side_effect = [producer_utils, producer_utils, consumer_utils, consumer_utils]

    producer_src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=_make_metadata(),
    )
    consumer_src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=_make_metadata(),
    )

    assert producer_src == consumer_src
    assert "/producer/cache" not in producer_src
    assert "/consumer/cache" not in consumer_src
    assert '#include <cstdlib>' in producer_src
    assert 'const char* cache_root = std::getenv("TRITON_CACHE_DIR");' in producer_src
    assert f'npu_utils_path = std::string(cache_root) + "/{cache_key}/npu_utils.so";' in producer_src
    assert 'const char* triton_home = std::getenv("TRITON_HOME");' in producer_src
    assert f'npu_utils_path = std::string(base) + "/.triton/cache/{cache_key}/npu_utils.so";' in producer_src
    module_init = producer_src.split("PyMODINIT_FUNC PyInit___triton_launcher", maxsplit=1)[1]
    assert "init_npu_utils();" in module_init
    assert "set_npu_utils_path" not in producer_src


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_rejects_unmatched_coalescing_metadata(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.coalesce_factor = 16
    metadata.coalesce_axis = 1

    with pytest.raises(RuntimeError, match="unmatched RowCoalescing"):
        driver.make_launcher(
            constants={},
            signature={0: "*fp32", 1: "*fp32"},
            metadata=metadata,
        )


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_uses_ceil_div_for_row_coalescing(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.coalesce_factor = 4
    metadata.coalesce_axis = 2
    metadata.coalesce_grid_ceil_div = True
    metadata.row_coalescing_applied = True
    metadata.auto_blockify_enabled = True

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    assert src.count("gridZ = (gridZ + 4 - 1) / 4;") == 2
    assert "ChunkCoalescing: grid[2] not divisible" not in src
    assert src.count("blockNum = std::min(blockNum, (uint32_t)40);") == 2


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_allows_auto_blockify_with_iat_mapping(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.program_grid_mapping_applied = True
    metadata.auto_blockify_enabled = True
    metadata.program_grid_transforms = _make_program_grid_contract({
        "order": 0,
        "kind": "ceil_div",
        "axis": 1,
        "factor": 16,
        "persistent_coverage": False,
        "grid_stride_abi_verified": False,
    })

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    assert src.count("gridY = (uint32_t)(((uint64_t)originalGridY + 15u) / 16u);") == 2
    assert src.count("blockNum = std::min(blockNum, (uint32_t)40);") == 2


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_enables_91095_simt_for_sls_mixed_parallel_mode(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.compile_on_910_95 = True
    metadata.parallel_mode = "mix_simd_simt"
    metadata.shared_mem_dynamic_size = 221184

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    assert src.count("cann_launch_kernel(func, blockNum") == 2
    assert src.count("cann_get_launch_kernel_cfg(221184)") == 2
    assert src.count("aclrtLaunchKernelWithHostArgs") == 1
    assert src.count("attrInfo.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;") == 1
    c_abi_launch, cpp_launch = _split_launch_functions(src)
    assert "static_cast<void*>(launch_args.data())" in c_abi_launch
    assert "&args" in cpp_launch
    assert "launch_args.size()" in c_abi_launch
    assert "sizeof(args)" in cpp_launch


@pytest.mark.parametrize(
    ("auto_blockify_enabled", "blacklisted"),
    (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ),
)
@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_block_cap_follows_compiler_auto_blockify_decision(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
    auto_blockify_enabled,
    blacklisted,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    cap = "blockNum = std::min(blockNum, (uint32_t)40);"

    for is_pure_simt, row_coalescing_applied in product(
        (False, True),
        (False, True),
    ):
        metadata = _make_metadata()
        metadata.has_auto_blockify_blacklist_op = blacklisted
        metadata.auto_blockify_enabled = auto_blockify_enabled
        metadata.is_pure_simt = is_pure_simt
        metadata.row_coalescing_applied = row_coalescing_applied
        if row_coalescing_applied:
            metadata.coalesce_factor = 4
            metadata.coalesce_axis = 0
            metadata.coalesce_grid_ceil_div = True
        src = driver.make_launcher(
            constants={},
            signature={0: "*fp32", 1: "*fp32"},
            metadata=metadata,
        )
        expected_per_launch_path = 1 if auto_blockify_enabled else 0
        c_abi_launch, cpp_launch = _split_launch_functions(src)
        case = (f"P={is_pure_simt}, B={blacklisted}, R={row_coalescing_applied}, "
                f"A={auto_blockify_enabled}")
        assert c_abi_launch.count(cap) == expected_per_launch_path, case
        assert cpp_launch.count(cap) == expected_per_launch_path, case


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_make_launcher_disables_block_cap_for_ptsm(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.program_grid_mapping_applied = True
    metadata.ptsm_cap_authorized = True
    metadata.program_grid_transforms = _make_program_grid_contract({
        "order": 0,
        "kind": "ceil_div",
        "axis": 0,
        "factor": 64,
        "persistent_coverage": True,
        "grid_stride_abi_verified": True,
    })

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    assert "blockNum = std::min(blockNum, (uint32_t)40);" not in src


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_merged_code_workspace_allocation_appears_in_both_paths(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.workspace_size = 1024

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    c_abi_launch, cpp_launch = _split_launch_functions(src)

    assert c_abi_launch.count("allocate_memory") == 1
    assert cpp_launch.count("allocate_memory") == 1
    assert c_abi_launch.count("workspace_handle_guard") == 1
    assert cpp_launch.count("workspace_handle_guard") == 1


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_merged_code_sync_block_lock_appears_in_both_paths(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20
    metadata = _make_metadata()
    metadata.sync_block_lock_layout = 2

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    c_abi_launch, cpp_launch = _split_launch_functions(src)

    assert c_abi_launch.count("/* allocate_sync_block_lock: ('syncBlockLockSize', 'stream') */") == 1
    assert cpp_launch.count("/* allocate_sync_block_lock: ('syncBlockLockSize', 'stream') */") == 1
    assert c_abi_launch.count("syncBlockLock_handle_guard") == 1
    assert cpp_launch.count("syncBlockLock_handle_guard") == 1


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_merged_code_msprof_calls_in_both_paths(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32"},
        metadata=_make_metadata(),
    )

    c_abi_launch, cpp_launch = _split_launch_functions(src)

    assert c_abi_launch.count("beginTime = MsprofSysCycleTime();") == 1
    assert cpp_launch.count("beginTime = MsprofSysCycleTime();") == 1
    assert c_abi_launch.count("MsprofReportApi(false, &info);") == 1
    assert cpp_launch.count("MsprofReportApi(false, &info);") == 1


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_merged_code_preamble_shared_variables_present(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32", 2: "i32"},
        metadata=_make_metadata(),
    )

    c_abi_launch, cpp_launch = _split_launch_functions(src)

    for section_name, section in [("triton_launch_kernel", c_abi_launch), ("_launch", cpp_launch)]:
        assert "void* workspace_addr_ptr = nullptr;" in section, f"{section_name}: missing workspace_addr_ptr"
        assert "void* workspace_handle = nullptr;" in section, f"{section_name}: missing workspace_handle"
        assert "uint32_t blockNum4Workspace = gridX * gridY * gridZ;" in section, f"{section_name}: missing blockNum4Workspace"
        assert "uint32_t blockNum = gridX * gridY * gridZ;" in section, f"{section_name}: missing blockNum"
        assert "cann_error ret = CANN_SUCCESS;" in section, f"{section_name}: missing ret"


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_merged_code_taskqueue_mode_in_both_paths(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32"},
        metadata=_make_metadata(),
    )

    c_abi_launch, cpp_launch = _split_launch_functions(src)

    assert c_abi_launch.count("std::function<cann_error()> launch_call") == 1
    assert cpp_launch.count("std::function<cann_error()> launch_call") == 1
    # The dlsym helpers in cpp_npu_utils_dlopen reference "async_launch" multiple
    # times (typedef, static decl, dlsym). Use the call site marker returned by
    # the mocked get_backend_func to verify the actual call appears in both paths.
    assert c_abi_launch.count("/* async_launch: ('launch_call',) */") == 1
    assert cpp_launch.count("/* async_launch: ('launch_call',) */") == 1
    assert c_abi_launch.count("return ret;") >= 1
    assert cpp_launch.count("return ret;") >= 1


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_merged_code_grid_warning_in_both_paths(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    with patch.dict("os.environ", {"TRITON_GRID_WARN_PRINT": "true"}):
        src = driver.make_launcher(
            constants={},
            signature={0: "*fp32", 1: "*fp32"},
            metadata=_make_metadata(),
        )

    c_abi_launch, cpp_launch = _split_launch_functions(src)

    assert c_abi_launch.count("Grid %u > physical limit") == 1
    assert cpp_launch.count("Grid %u > physical limit") == 1


@patch.object(driver, "NPUUtils")
@patch.object(driver, "force_disable_ffts", return_value=False)
@patch.object(driver, "is_ffts_supported", return_value=True)
@patch.object(driver, "get_backend_func", side_effect=_mock_backend_func)
def test_argument_packing_differs_between_launch_paths(
    _mock_backend_func_patch,
    _mock_ffts,
    _mock_disable_ffts,
    mock_npu_utils,
):
    mock_npu_utils.return_value.get_aivector_core_num.return_value = 40
    mock_npu_utils.return_value.get_aicore_num.return_value = 20

    src = driver.make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32", 2: "i32"},
        metadata=_make_metadata(),
    )

    c_abi_launch, cpp_launch = _split_launch_functions(src)

    assert "std::vector<char> launch_args(total_size, 0);" in c_abi_launch
    assert "reserve_slot" in c_abi_launch

    assert "struct __attribute__((packed))" in cpp_launch
    assert "void* ffts_addr __attribute__((aligned(8)));" in cpp_launch
    assert "memcpy(launch_args.data()" not in cpp_launch


@patch("importlib.util.module_from_spec")
@patch("importlib.util.spec_from_file_location")
@patch.object(driver, "make_npu_launcher_stub", return_value="/tmp/fake_launcher.so")
@patch.object(driver, "make_launcher", return_value="// wrapper src")
@patch.object(driver, "generate_npu_header_src", return_value="// header src")
def test_npu_launcher_exposes_launcher_so_path(
    mock_header_src,
    mock_wrapper_src,
    mock_launcher_stub,
    mock_spec_from_file_location,
    mock_module_from_spec,
):
    fake_module = SimpleNamespace(launch=object())
    fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    mock_module_from_spec.return_value = fake_module
    mock_spec_from_file_location.return_value = fake_spec
    src = SimpleNamespace(
        constants={"input_ptr": 1},
        signature={"input_ptr": "*fp32", "numel": "i32"},
        fn=SimpleNamespace(arg_names=["input_ptr", "numel"]),
    )
    metadata = _make_metadata()

    launcher = driver.NPULauncher(src, metadata)

    assert launcher.so_launcher_path == "/tmp/fake_launcher.so"
    assert mock_launcher_stub.call_count == 1
    assert launcher.get_launcher_so_path() == "/tmp/fake_launcher.so"
    assert mock_header_src.call_count == 1
    assert mock_wrapper_src.call_count == 1
    assert mock_launcher_stub.call_count == 1
    mock_wrapper_src.assert_called_with(
        {0: 1},
        {0: "*fp32", 1: "i32"},
        metadata,
    )
    mock_launcher_stub.assert_called_with("// header src", "// wrapper src", False)
