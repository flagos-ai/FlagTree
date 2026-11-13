"""Ascend tool and consumer wheel staging; loaded without importing triton.

This module is loaded by file path so the legacy build package __init__ and
its setup patch machinery are not executed.
"""
import os
import subprocess
from pathlib import Path
import shutil


def _build_npuir_enabled():
    value = os.environ.get("TRITON_BUILD_NPUIR", "OFF").strip().upper()
    if value in {"ON", "1", "YES", "TRUE", "Y"}:
        return True
    if value in {"OFF", "0", "NO", "FALSE", "N"}:
        return False
    raise RuntimeError(f"Invalid TRITON_BUILD_NPUIR value: {value!r}")


def _remove_generated_directory(path):
    if path.is_symlink():
        raise RuntimeError(f"Refusing to clean symlinked build directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Expected a generated directory: {path}")
        shutil.rmtree(path)


def _copy_ascend_tools(extdir, cmake_dir):
    """Mirror triton-ascend's copy-then-strip hook; retain build-tree symbols."""
    source_dir = Path(cmake_dir) / "third_party/ascend/bin"
    destination_dir = Path(extdir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in ("triton-opt", "triton-mlir-opt"):
        source = source_dir / name
        if not source.is_file():
            print(f"[ascend] {name} was not built; skipping", flush=True)
            continue
        destination = destination_dir / name
        shutil.copy2(source, destination)
        destination.chmod(0o755)
        print(f"[ascend] Copied {name} to {destination}", flush=True)
        try:
            subprocess.check_call(["strip", "--strip-all", str(destination)])
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            print(f"[ascend] WARNING: could not strip {name}: {error}", flush=True)
        else:
            print(f"[ascend] Stripped {name} to reduce size", flush=True)


def _validate_bishengir_payload(payload):
    for name in ("bishengir-opt", "bishengir-compile", "hivmc", "hivmc-a5"):
        executable = payload / "bin" / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Missing BiShengIR executable: {executable}")
    # Template bitcode is matched by stem rather than by exact file name: an
    # AscendNPU-IR revision that supports --enable-optimized-metaop emits
    # per-optimization-level variants (meta_op.aic.c220.O2.bc) instead of the
    # single meta_op.aic.c220.bc, and both layouts are valid payloads.
    stems = [f"meta_op.{kind}.{arch}" for arch in ("c220", "c310")
             for kind in ("aic", "aiv", "mix.aic", "mix.aiv")]
    for stem in (*stems, "host"):
        found = [path for path in payload.glob(f"lib/{stem}.*bc") if path.stat().st_size]
        if not found:
            raise RuntimeError(f"Missing or empty BiShengIR template: {payload / 'lib' / (stem + '*.bc')}")


def _build_bishengir():
    # setup.py loads the registry under python.setup_tools, not setup_tools.
    from python.setup_tools.utils import get_submodules

    module = get_submodules("AscendNPU-IR")
    source = Path(module.dst_path).resolve()
    script = source / "build-tools" / "build.sh"
    if not script.is_file():
        raise RuntimeError(f"AscendNPU-IR source is not prepared: {source}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != module.commit_id:
        raise RuntimeError(f"AscendNPU-IR is at {head}, expected {module.commit_id}; prepare the pinned checkout")
    cann_home = os.environ.get("ASCEND_HOME_PATH")
    if not cann_home:
        raise RuntimeError("TRITON_BUILD_NPUIR=ON requires ASCEND_HOME_PATH; source CANN set_env.sh first")
    bisheng_bin = Path(cann_home).expanduser().resolve() / "tools" / "bisheng_compiler" / "bin"
    for name in ("ccec", "llvm-link"):
        executable = bisheng_bin / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Missing CANN build tool: {executable}")
    for name in ("git", "cmake", "ninja", "clang", "clang++"):
        if shutil.which(name) is None:
            raise RuntimeError(f"Required build tool is not on PATH: {name}")
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("LLVM_", "MLIR_", "CMAKE_")):
            env.pop(name)
    for name in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS"):
        env.pop(name, None)
    subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=source, env=env)
    build_root = source / "build"
    _remove_generated_directory(build_root)
    build_root.mkdir()
    payload = build_root / "bishengir-output"
    cmake_options = ["-DLLVM_ENABLE_LLD=" + ("ON" if shutil.which("ld.lld") else "OFF")]
    if shutil.which("ccache"):
        cmake_options.extend(["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"])
    command = [
        "bash", str(script), "--build-type=Release", "-o", str(build_root / "bishengir"),
        "-t", "-j", "32", "-r", f"--bisheng-compiler={bisheng_bin}",
        "--build-triton", "--build-torch-mlir", "--build-shmem-template",
        "--bishengir-publish", "ON", "--collect-binary", str(payload),
        "--add-cmake-options", " ".join(cmake_options),
    ]
    print(f"[ascend] Building BiShengIR under {build_root}", flush=True)
    subprocess.check_call(command, cwd=source, env=env)
    _validate_bishengir_payload(payload)
    return payload


def finalize_build_tree(build_lib):
    enabled = _build_npuir_enabled()
    build_root = Path(build_lib).resolve()
    destination = build_root / "triton" / "backends" / "ascend" / "bishengir"
    if not destination.parent.resolve().is_relative_to(build_root):
        raise RuntimeError(f"Backend staging escapes build_lib: {destination}")
    _remove_generated_directory(destination)
    if not enabled:
        return
    payload = _build_bishengir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(payload, destination, copy_function=shutil.copy2)
    _validate_bishengir_payload(destination)
    print(f"[ascend] Bundled BiShengIR into {destination}", flush=True)
