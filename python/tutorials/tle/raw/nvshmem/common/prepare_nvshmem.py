#!/usr/bin/env python3

import argparse
import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path


NVSHMEM_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_FILE = Path(__file__).with_name("generate_extern_call.py")
_SPEC = importlib.util.spec_from_file_location("_tle_generate_extern_call", GENERATOR_FILE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load extern call generator from {GENERATOR_FILE}")
_GENERATOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GENERATOR
_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _SPEC.loader.exec_module(_GENERATOR)
finally:
    sys.dont_write_bytecode = _DONT_WRITE_BYTECODE
generate = _GENERATOR.generate


def _last_string(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    for child in reversed(list(ast.iter_child_nodes(node))):
        value = _last_string(child)
        if value is not None:
            return value
    return None


def _dialect_file_pairs(example_dir):
    pairs = set()
    for python_file in example_dir.glob("*.py"):
        tree = ast.parse(python_file.read_text(encoding="utf-8"), filename=str(python_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "dialect":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            cuda_name = _last_string(keywords.get("file")) if "file" in keywords else None
            extern_name = _last_string(keywords.get("extern")) if "extern" in keywords else None
            if cuda_name and extern_name:
                pairs.add((example_dir / cuda_name, example_dir / extern_name))
    return pairs


def generate_extern_files(example_dir):
    pairs = _dialect_file_pairs(example_dir)
    generated = []
    for cuda_file, output_file in sorted(pairs, key=lambda pair: str(pair[1])):
        if not cuda_file.is_file():
            raise FileNotFoundError(f"CUDA device file does not exist: {cuda_file}")
        generate(cuda_file, output_file)
        generated.append(output_file)
    return generated


def detect_arch(explicit_arch):
    arch = explicit_arch or "sm_80"
    arch = arch.removeprefix("-arch=").removeprefix("sm_")
    return f"sm_{arch}"


def resolve_nvshmem_home(explicit_home):
    if not explicit_home:
        raise ValueError(
            "NVSHMEM_HOME is required; set the environment variable or pass --nvshmem-home"
        )
    home = Path(explicit_home).expanduser().resolve()

    if not (home / "include").is_dir() or not (home / "lib").is_dir():
        raise FileNotFoundError(
            f"invalid NVSHMEM_HOME {home}: expected include/ and lib/ directories"
        )
    return home


def compile_host_files(example_dir, nvshmem_home, arch, force):
    nvcc = os.getenv("NVCC", "nvcc")
    outputs = []
    for cuda_file in sorted(example_dir.glob("*-host.cu")):
        output_file = cuda_file.with_suffix(".so")
        if (
            not force
            and output_file.exists()
            and output_file.stat().st_mtime_ns >= cuda_file.stat().st_mtime_ns
        ):
            outputs.append(output_file)
            continue

        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            dir=output_file.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        command = [
            nvcc,
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-rdc=true",
            f"-arch={arch}",
            f"-I{nvshmem_home / 'include'}",
            f"-L{nvshmem_home / 'lib'}",
            "-lnvshmem_host",
            "-lnvshmem_device",
            "-o",
            str(temporary_path),
            str(cuda_file),
        ]
        try:
            subprocess.run(command, check=True)
            os.replace(temporary_path, output_file)
        finally:
            temporary_path.unlink(missing_ok=True)
        outputs.append(output_file)
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate Triton extern calls and compile NVSHMEM host libraries."
    )
    parser.add_argument("target", type=Path, help="NVSHMEM example Python file")
    parser.add_argument("--nvshmem-home", default=os.getenv("NVSHMEM_HOME"))
    parser.add_argument("--arch", default="sm_80", help="CUDA architecture (default: sm_80)")
    parser.add_argument("--force", action="store_true", help="always rebuild host libraries")
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    if not target.is_file():
        parser.error(f"target does not exist: {target}")
    if NVSHMEM_ROOT not in target.parents:
        parser.error(f"target must be below {NVSHMEM_ROOT}")

    nvshmem_home = resolve_nvshmem_home(args.nvshmem_home)
    arch = detect_arch(args.arch)
    generated = generate_extern_files(target.parent)
    libraries = compile_host_files(target.parent, nvshmem_home, arch, args.force)

    for path in generated:
        print(f"[prepare] extern: {path}")
    for path in libraries:
        print(f"[prepare] host library: {path}")
    print(f"[prepare] NVSHMEM_HOME={nvshmem_home}")
    print(f"[prepare] CUDA architecture={arch}")


if __name__ == "__main__":
    main()
