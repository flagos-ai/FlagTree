import os

lit_config.load_config(
    config,
    os.path.join(config.triton_src_root, "test", "lit.cfg.py"),
)
config.name = "FLAGTREE-DEBUGGER"
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.triton_obj_root, "third_party", "Debugger", "test", "lit")

filecheck = lit_config.params.get("filecheck")
if filecheck and os.path.isfile(filecheck):
    filecheck_command = filecheck
else:
    fallback = os.path.join(os.path.dirname(__file__), "Inputs", "filecheck.py")
    filecheck_command = f'"{config.python_executable}" "{fallback}"'
config.substitutions.append(("FileCheck", filecheck_command))
not_fallback = os.path.join(os.path.dirname(__file__), "Inputs", "not.py")
config.substitutions.append(("not ", f'"{config.python_executable}" "{not_fallback}" '))

config.excludes = set(config.excludes)
config.excludes.update([
    "simplify-record-memref-writes.mlir",
    "statement-operand-capture-example.mlir",
])
