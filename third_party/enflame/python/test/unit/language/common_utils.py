import pytest
import csv
import re
import os
import triton
import importlib.util
if importlib.util.find_spec("triton.backends.enflame") is None:
    import triton_gcu.triton


def check_skip(file_path, test_name, params=None):
    """
    Checks if a test case should be skipped based on data from 'skip_tests_write.csv'.
    Args:
        file_path (str): relative path of csv file
        test_name (str): the name of the test function
        params (dict, optional): input params to the test function
    Usage:
        import inspect

        def test_func():
            check_skip("skip.csv", inspect.currentframe().f_code.co_name, locals())
    """
    if params is None:
        params = {}

    # Remove 'device' key if present
    if (test_name != "test_pointer_arguments" or params["device"] == "cuda"):
        params.pop("device", None)

    if "tmp_path" in params:
        params.pop("tmp_path", None)

    if "test_gather_warp_shuffle" in test_name:
        params.pop("src_layout", None)
        params.pop("indices_layout", None)

    # for test_convertmma2mma convert mma_pair object to parseable string
    if "mma_pair" in params:
        params["mma_pair"] = [
            re.sub(r"versionMajor=\d+,?\s*|versionMinor=\d+,?\s*", "", str(x)) for x in params["mma_pair"]
        ]

    target_arch = triton.runtime.driver.active.get_current_target().arch.split("--")[1]

    # Generate a unique case identifier
    case_identifier = (test_name + "_" + "".join(f"{key}={value}-" for key, value in params.items()))

    try:
        print(f"\nDEBUG(check_skip): Case Identifier: <{case_identifier}>\n")
        if (target_arch == 'gcu400' or target_arch == 'gcu410'):
            if ("num_warps" in params and params["num_warps"] == 8):
                pytest.skip("num_warps 8 is not supported in gcu4xx")
        # Read the CSV file and check if the test case should be skipped
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)

        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not (target_arch in row["arch"].split("/")):
                    continue
                if row["case_identifier"] == case_identifier:
                    # Construct skip message
                    reason = row["resb"] if row["resb"] else "Failed on gcu"
                    skip_message = f"{reason}: {row['resa']}"
                    print("Skipping with message: ", skip_message)

                    pytest.skip(skip_message)

    except FileNotFoundError:
        print("Error: 'test_core_skip.csv' not found.")
    except KeyError as e:
        print(f"Missing expected column in CSV: {e}")
