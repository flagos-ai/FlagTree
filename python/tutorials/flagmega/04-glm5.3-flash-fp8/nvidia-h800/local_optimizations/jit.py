"""Compile all real entry ABIs without loading weights or launching kernels."""
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import os
HERE = Path(__file__).resolve().parent.parent
_SDK = Path(os.environ['FLAGCX_MODULE_PATH']).resolve()
_dependencies = json.loads((HERE/'local_optimizations/reproduction-runtime-dependencies.json').read_text())['sdk']
assert hashlib.sha256((_SDK/'libflagcx_device.bc').read_bytes()).hexdigest() == _dependencies['device_bc_sha256']
# Host ELF metadata depends on the build directory. JIT uses the device
# bitcode; the host protocol is validated by the separate full runtime run.
_sdk_host_sha256 = hashlib.sha256((_SDK/'libflagcx.so').read_bytes()).hexdigest()

import torch
from triton.flagmega.runtime.module import _resource_contract, _torch_dtype
from triton.flagmega.runtime.nvidia_launch import compilation_options
from triton.flagmega.runtime.prepared import _spill_resource_report, _validate_resources
from triton.flagmega.runtime.program_dag import verify_joint_resources
from triton.runtime import DistributedRtContext

OUT = HERE
package = json.loads((HERE/"artifact/artifact.json").read_text())["codegen"]
source = HERE/"artifact/generated_kernels.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == package["source_sha256"]
spec = importlib.util.spec_from_file_location("_flagmega_shared_program_jit", source)
generated = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generated
spec.loader.exec_module(generated)
DistributedRtContext._reset()

def scalar(dtype):
    return False if dtype == "bool" else 0. if dtype.startswith("float") or dtype == "bfloat16" else 0

roots = {}
for arg in package["runtime_binding"]["arguments"]:
    roots[arg["name"]] = scalar(arg["scalar_dtype"]) if arg.get("runtime_value_kind") == "scalar" else torch.empty(256, device="cuda", dtype=torch.uint8)
for pool in package["runtime_binding"]["pools"]:
    roots[pool["name"]] = torch.empty(256, device="cuda", dtype=torch.uint8)
for table in package["function_offset_tables"]:
    roots[table["name"]] = torch.tensor(table["values"], device="cuda", dtype=torch.int64)
for desc in package["host_tensor_descriptor_specs"]:
    assert desc.get("storage") == "device" or desc["kind"] == "table", desc
    roots[desc["name"]] = torch.empty(128, device="cuda", dtype=torch.uint8)
roots.update(device_mem_ptr=0x7F9EBF201000, device_comm_ptr=0x7F9EBF200E00, device_context_mode=False)
# Unique primitive entries retain their original external pointer types.
from triton.flagmega import ir as fm
sys.path.insert(0, str(HERE))
from local_optimizations.kernel_recipe import activate
activate()
module = fm.load_module(OUT / "dag-final.py")
plan = fm.verify_buffer_plan(module)
for arg in package["runtime_binding"]["arguments"]:
    if arg.get("runtime_value_kind") != "scalar":
        roots[arg["name"]] = torch.empty(256, device="cuda", dtype=_torch_dtype(torch, plan.buffer_map[arg["buffer"]].dtype))
root_values = [roots[n] for n in package["signature_arguments"]]
report = dict(sdk_host_sha256=_sdk_host_sha256,
              sdk_device_bc_sha256=_dependencies['device_bc_sha256'],
              source_sha256=package["source_sha256"], semantic_hash=module.semantic_hash,
              diagnostic_only=True, no_weights_loaded=True, kernels_never_launched=True,
              invocations=len(package["kernels"]), entries=[], jit_ok=False, joint_resources_ok=False)
started = time.perf_counter()
prepared, unique, compiled_entries = [], {}, {}
try:
    for spec in package["kernels"]:
        kernel = getattr(generated, spec["symbol"])
        arguments = []
        for binding in spec["argument_bindings"]:
            if binding["kind"] == "immediate":
                value = binding["value"]
            elif binding["kind"] == "pointer":
                # Only pointer type/alignment are consumed by warmup=True.
                # Actual addresses/offsets are checked by normal runtime tests.
                value = torch.empty(256, device="cuda", dtype=getattr(torch, binding["dtype"]))
            else:
                value = root_values[binding["source"]]
            arguments.append(value)
        contract = _resource_contract({**package, "num_warps": spec["num_warps"]}, "nvidia-sm90",
            requires_grid_wide_sync=spec["requires_grid_wide_sync"], device="cuda:0")
        begin = time.perf_counter()
        compiled = kernel.run(*arguments, grid=tuple(spec["grid"]), warmup=True, **compilation_options(contract))
        if hasattr(compiled, "result"):
            compiled = compiled.result()
        _validate_resources(compiled, contract, tuple(spec["grid"]))
        resources = dict(name=compiled.name, num_warps=compiled.metadata.num_warps,
            registers_per_thread=compiled.n_regs, shared_memory_bytes=compiled.metadata.shared,
            **_spill_resource_report(compiled, contract))
        identity = compiled_entries.setdefault(spec["entry_symbol"], compiled.hash)
        assert identity == compiled.hash, ("Unexpected per-invocation specialization", spec["symbol"], identity, compiled.hash)
        prepared.append(SimpleNamespace(compiled_kernel=compiled, contract=contract, resource_report=resources))
        if compiled.hash not in unique:
            record = dict(symbol=spec["symbol"], function=spec["entry_function"], hash=compiled.hash,
                          seconds=time.perf_counter()-begin, **resources)
            unique[compiled.hash] = record
            report["entries"].append(record)
            print(json.dumps(record), flush=True)
    report["jit_ok"] = True
    report["unique_compiled_entries"] = len(unique)
    report["joint_resources"] = verify_joint_resources(package["kernels"], prepared,
        torch.cuda.get_device_properties(0).multi_processor_count)
    report["joint_resources_ok"] = True
except BaseException:
    report["traceback"] = traceback.format_exc()
    raise
finally:
    report["seconds"] = time.perf_counter()-started
    (OUT / "warmup.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
