#
# Copyright 2024 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from pathlib import Path
import subprocess

device_name = "gcu"
datadir = "/opt/triton_gcu"
if not os.path.exists(datadir):
    raise Exception("Cannot find data directory in " + datadir)

TOOLKIT_PATH = os.path.join(datadir, "bin")
RUNTIME_PATH = os.path.join(datadir, "lib")

PY_TOOLS_PATH = Path(__file__).parent


# toolkit
def _run_command(cmd, content, *args):
    if not isinstance(content, str):
        content = str(content)
    result = subprocess.run([os.path.join(TOOLKIT_PATH, cmd)] + list(args), input=content, capture_output=True,
                            text=True, encoding="utf-8")
    if result.returncode != 0:
        raise Exception(result.stderr)
    # print(__file__, "run command: \n", [os.path.join(TOOLKIT_PATH, cmd)] + list(args))
    # print subprocess std::cerr << log to terminator
    print(result.stderr)
    return result.stdout


def _run_command2(cmd, content, *args):
    if not isinstance(content, str):
        content = str(content)
    result = subprocess.run([PY_TOOLS_PATH / cmd] + list(args), input=content, capture_output=True, text=True,
                            encoding="utf-8")
    if result.returncode != 0:
        raise Exception(result.stderr)
    # print(__file__, "run command: \n", [os.path.join(PY_TOOLS_PATH, cmd)] + list(args))
    # print(subprocess.stderr)
    print(result.stderr)
    return result.stdout


def triton_gcu_opt(content, *args, arch):
    passes = ["-mlir-print-op-generic"] + list(args)
    if arch == "gcu410":
        arch = "gcu400"
    return _run_command2(f"triton-{arch}-opt", content, *passes)


def gcu_compiler_opt(content, *args):
    passes = ["-mlir-print-op-generic"] + list(args)
    return _run_command("gcu-compiler-opt", content, *passes)


def compile(content, *args):
    return _run_command("gcu-compiler-compile", content, *args)


# Return the boolean value of an environment variable.
#
# Helpful environment variables:
#
# - "MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs and
# the IR after every MLIR pass GCU runs.
def get_bool_env(env, defaultValue=False):
    s = os.getenv(env, "").lower()
    if (s == "1" or s == "true" or s == "on"):
        return True
    if (s == "0" or s == "false" or s == "off"):
        return False
    return defaultValue
