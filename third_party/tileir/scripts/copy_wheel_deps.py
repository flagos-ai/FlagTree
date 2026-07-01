#!/usr/bin/env python3
"""
Copy cuda_dep tree into third_party/tileir/backend so the wheel packs it.
Preserves cuda_dep layout (bin/, nvvm/lib64/, etc.) so CUDA_HOME can point at
backend/cuda_dep. Reads wheel_deps.json; ${arch} -> x86 or arm64.
Run before: python -m build --wheel
"""
import json
import os
import platform
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "third_party", "tileir", "backend")

ARCH = platform.machine()
DEP_ARCH = "x86" if ARCH in ("x86_64", "amd64") else "arm64" if ARCH in ("aarch64", "arm64") else None
if DEP_ARCH is None:
    sys.exit(f"Unsupported arch: {ARCH}")

with open(os.path.join(SCRIPT_DIR, "wheel_deps.json")) as f:
    deps = json.load(f)

for dest_name, path_tpl in deps.get("copy", {}).items():
    src_rel = path_tpl.replace("${arch}", DEP_ARCH)
    src = os.path.join(SCRIPT_DIR, src_rel)
    dst = os.path.join(BACKEND_DIR, dest_name)
    if not os.path.isdir(src):
        sys.exit(f"Missing dir: {src}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # Ensure binaries are executable
    bin_dir = os.path.join(dst, "bin")
    if os.path.isdir(bin_dir):
        for name in os.listdir(bin_dir):
            p = os.path.join(bin_dir, name)
            if os.path.isfile(p):
                os.chmod(p, 0o755)
