import subprocess
import sys

from pathlib import Path
import hashlib


def _cache_key(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_command(command):
    result = subprocess.run(command, text=True, stderr=subprocess.PIPE)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        detail = result.stderr.strip() or "no diagnostic was emitted"
        raise RuntimeError(f"Command '{' '.join(command)}' failed:\n{detail}")
    print(f"Command '{' '.join(command)}' executed successfully.")


def compile_to_linalg(jit_func, *args, **kwargs):
    kwargs["debug"] = None
    kwargs["to_ttsharedir"] = True
    return jit_func.warmup(*args, grid=[
        1,
    ], **kwargs)


def ttsharedir_compare(jit_func, *args, golden_ir_dir, **kargs):
    ttshared_kernel = compile_to_linalg(jit_func, *args, **kargs)
    golden_ir_file = Path(golden_ir_dir) / (ttshared_kernel.name + ".mlir")
    if golden_ir_file.exists() and golden_ir_file.is_file():
        if golden_ir_file.read_bytes() == ttshared_kernel.asm["elf"]:
            print(f"Compiled IR compared with {golden_ir_file} successfully.")
            return True
        raise AssertionError(f"compiled IR compared with {golden_ir_file} failed")
    raise LookupError(f"{golden_ir_file} is not found")
