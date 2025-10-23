import os
import subprocess
import re
import pytest


# 执行test_device_print.py -> 生成 test_device_print.log, 以便执行后续验证
def run_test_device_print_py(test_name, log_name):
    testfile_path = os.path.join(os.getcwd(), test_name)
    logfile_path = os.path.join(os.getcwd(), log_name)

    with open(logfile_path, 'w') as f:
        try:
            subprocess.run(["pytest", testfile_path], stdout=f, stderr=subprocess.STDOUT, check=True)
            print(f"Run 【{test_name}】 successfully!")
        except Exception as e:
            print(f"Run 【{test_name}】 unsuccessfully: ", e)


def assert_close(expected_output, logfile):
    # 读回日志内容
    with open(logfile, "r", encoding="utf-8") as f:
        raw = f.read()
        cleaned = re.sub(r"\x00", "", raw)

    # 在文件内容里做检查
    assert expected_output in cleaned, f"Expected '{expected_output}' not found in log file."


@pytest.mark.skip(reason="waiting for TA to support")
def test_device_print_int8():
    expected_output = "0,-128,127,0,-1,0,-1,0"
    test_name = "test_device_print.py::test_device_print_int8[int8]"
    log_name = "test_device_print_int8.log"
    logfile = os.path.join(os.getcwd(), log_name)
    run_test_device_print_py(test_name, logfile)
    assert_close(expected_output, logfile)


@pytest.mark.skip(reason="waiting for TA to support")
def test_device_print_int16():
    expected_output = "0,-128,127,-32768,32767,0,-1,0"
    test_name = "test_device_print.py::test_device_print_int16[int16]"
    log_name = "test_device_print_int16.log"
    logfile = os.path.join(os.getcwd(), log_name)
    run_test_device_print_py(test_name, logfile)
    assert_close(expected_output, logfile)


@pytest.mark.skip(reason="waiting for TA to support")
def test_device_print_int32():
    expected_output = "0,-128,127,-32768,32767,-2147483648,2147483647,-2147483648"
    test_name = "test_device_print.py::test_device_print_int32[int32]"
    log_name = "test_device_print_int32.log"
    logfile = os.path.join(os.getcwd(), log_name)
    run_test_device_print_py(test_name, logfile)
    assert_close(expected_output, logfile)


@pytest.mark.skip(reason="waiting for compiler to support")
def test_device_print_int64():
    expected_output = "???"
    test_name = "test_device_print.py::test_device_print_int64[int64]"
    log_name = "test_device_print_int64.log"
    logfile = os.path.join(os.getcwd(), log_name)
    run_test_device_print_py(test_name, logfile)
    assert_close(expected_output, logfile)


@pytest.mark.skip(reason="waiting for TA to support")
def test_device_print_fp16():
    expected_output = "0.000000,0.000000,0.000977,0.007812,inf,65504.000000,inf,1.000000"
    test_name = "test_device_print.py::test_device_print_fp16[float16]"
    log_name = "test_device_print_fp16.log"
    logfile = os.path.join(os.getcwd(), log_name)
    run_test_device_print_py(test_name, logfile)
    assert_close(expected_output, logfile)


@pytest.mark.skip(reason="waiting for TA to support")
def test_device_print_fp32():
    expected_output = "0.000000,0.000000,0.000977,0.007812,340282346638528859811704183484516925440.000000,65504.000000,338953138925153547590470800371487866880.000000,1.000000"
    test_name = "test_device_print.py::test_device_print_fp32[float32]"
    log_name = "test_device_print_fp16.log"
    logfile = os.path.join(os.getcwd(), log_name)
    run_test_device_print_py(test_name, logfile)
    assert_close(expected_output, logfile)


@pytest.mark.skip(reason="waiting for compiler to support")
def test_device_print_bf16():
    expected_output = "???"
    test_name = "test_device_print.py::test_device_print_bf16[bfloat16]"
    log_name = "test_device_print_bf16.log"
    logfile = os.path.join(os.getcwd(), log_name)
    run_test_device_print_py(test_name, logfile)
    assert_close(expected_output, logfile)