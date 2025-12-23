# content of conftest.py

import pytest

from score_listener import ResultsCollectorPlugin


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default='cuda')
    parser.addoption("--result_dir", type=str, default="./log_result")
    parser.addoption("--result_file", type=str, default="___test-summary-triton.json")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


def pytest_configure(config):
    config.pluginmanager.register(ResultsCollectorPlugin(config), 'results_collector')


# 需要加 marker 的文件列表
FILES_TO_MARK = [
    "language.test_annotations",
    "language.test_block_pointer",
    "language.test_compile_errors",
    "language.test_conversions",
    "language.test_core",
    "language.test_decorator",
    "language.test_iluvatar_bf16",
    "language.test_line_info",
    "language.test_random",
    "language.test_reproducer",
    "language.test_standard",
    "language.test_subprocess",
    "operators.test_blocksparse",
    "operators.test_cross_entropy",
    "operators.test_dot_trans",
    "operators.test_flash_attention",
    "operators.test_inductor",
    "operators.test_matmul",
    "runtime.test_autotuner",
    "runtime.test_bindings",
    "runtime.test_cache",
    "runtime.test_driver",
    "runtime.test_jit",
    "runtime.test_launch",
    "runtime.test_subproc",
    "operators.test_sme",
]

MARK_NAME = "level_0"


def pytest_collection_modifyitems(config, items):
    """
    给指定模块里的测试用例统一加 marker
    """
    for item in items:
        module_name = item.module.__name__
        if module_name in FILES_TO_MARK:
            item.add_marker(pytest.mark.__getattr__(MARK_NAME))
