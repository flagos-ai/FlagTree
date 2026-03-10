import binascii
import hashlib
import importlib.util
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import triton
import triton_gcu.triton


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "u1": "uint32_t",
        "i8": "int8_t",
        "u8": "uint8_t",
        "i16": "int16_t",
        "u16": "uint16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "f16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "index": "int64_t",
    }[ty]


def write_to_file(dir, file_name, src):
    path = os.path.join(dir, file_name)
    with open(path, "w") as file:
        file.write(src)

    return path


def _is_type_token(tok):
    """判断 signature 中的 token 是类型还是字面量（整数）。"""
    t = tok.strip()
    if t.startswith("*"):
        return True
    type_map = {
        "i1",
        "u1",
        "i8",
        "u8",
        "i16",
        "u16",
        "i32",
        "i64",
        "u32",
        "u64",
        "fp16",
        "f16",
        "bf16",
        "fp32",
        "f32",
        "fp64",
        "index",
    }
    return t in type_map


# TODO: consider GCU400 later on
def auto_generate_global_func_str(global_func_name, func_entry, func_signature):
    tokens = [t.strip() for t in func_signature.split(",")]
    param_names = [f"arg{i}" for i in range(len(tokens))]
    # 与 get_global_func_str 一致：device 只接收 3 个指针 + s0, num_tasks, num_tiles, tiles_per_cta（索引 0,1,2,9,10,11,12）
    device_param_indices = [0, 1, 2, 9, 10, 11, 12]
    decl_parts = []
    for i, tok in enumerate(tokens):
        name = param_names[i]
        if _is_type_token(tok):
            cpp_ty = ty_to_cpp(tok)
            decl_parts.append(f"{cpp_ty} {name}")
        else:
            decl_parts.append(f"int32_t {name}")
    decl_str = ",\n    ".join(decl_parts)
    device_decl_parts = []
    device_call_names = []
    for idx in device_param_indices:
        if idx < len(tokens):
            name = param_names[idx]
            tok = tokens[idx]
            if _is_type_token(tok):
                cpp_ty = ty_to_cpp(tok)
                device_decl_parts.append(f"{cpp_ty} {name}")
            else:
                device_decl_parts.append(f"int32_t {name}")
            device_call_names.append(name)
    device_decl_str = ",\n    ".join(device_decl_parts)
    device_call_str = ", ".join(device_call_names)
    global_func_src = f"""
#include <tops.h>
#include <tops/tops_runtime.h>

extern "C" {{

__device__ void {func_entry}(
    {device_decl_str});

__global__ __thread_dims__(1, 1, 1) void {global_func_name}(
    {decl_str}) {{
  {func_entry}({device_call_str});
}}

}}
"""
    return global_func_src


desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

The resulting entry point will have signature

CUresult kernel_{specialization_suffix}(CUstream stream, unsigned gX, unsigned gY, unsigned gZ, float* arg0, int32_t arg1, int32_t arg2)

Different such specialized entry points can be combined using the `linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `compile.py` script
"""

if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path", help="Input Path of Python source containing desired kernel in its scope.")
    parser.add_argument("--global-func-name", "-g", type=str, help="Name of the global function", required=True)
    parser.add_argument("--out-path", "-o", type=Path, help="Output Filename of object file", required=True)
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--kernel-name", "-n", type=str, help="Name of the kernel to compile", required=True)
    parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages", "-ns", type=int, default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--enable-i64-check", "-i64", type=int, default=0,
                        help="Enable int64 type in triton_gcu of gcu300")
    args = parser.parse_args()

    # execute python sources and extract functions wrapped in JITFunction
    os.environ["COMPILE_AOT"] = "1"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    os.environ["ENABLE_I64_CHECK"] = "0" if args.enable_i64_check == 0 else "1"
    os.environ["AOT_OUT_PATH"] = str(Path(args.out_path))
    out_path_dir = os.path.dirname(args.out_path)
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    global_func_name = args.global_func_name
    global_func_src = auto_generate_global_func_str(global_func_name, args.kernel_name, args.signature)
    os.environ["GLOBAL_FUNC_PATH"] = write_to_file(out_path_dir, f"{global_func_name}.tops", global_func_src)
    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        signature[key] = 'constexpr'
    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
    opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    ccinfo = triton.compile(src, options=opts)
