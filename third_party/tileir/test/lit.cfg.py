# ruff: noqa: F821
import os
import lit.formats
from lit.llvm import llvm_config

config.name = "TRITON-CUDA-TILE"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.excludes = ["Inputs"]
# Use only the tools shipped in Triton's LLVM package.
llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
llvm_config.with_environment("FILECHECK_OPTS", "--enable-var-scope")
tool_dirs = [config.tileir_tools_dir, config.triton_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(["triton-cuda-tile-opt", "triton-opt", "FileCheck"], tool_dirs)
