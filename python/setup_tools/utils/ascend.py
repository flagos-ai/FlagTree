import importlib.util
import os
import subprocess
from pathlib import Path


def install_extension(*args, **kargs):
    """Copy producer tools next to libtriton, then strip only the copies."""
    from python.build_helpers import get_cmake_dir

    build_ext = kargs.get("build_ext") or (args[0] if args else None)
    if build_ext is None:
        raise RuntimeError("Ascend install_extension requires the build_ext command")

    # Match setup.py's CMakeBuild.build_extension: ext.path, not ext.name.
    ext = next(ext for ext in build_ext.extensions if ext.name == "triton")
    extdir = Path(build_ext.get_ext_fullpath(ext.path)).resolve().parent
    source = Path(__file__).resolve().parents[3] / "third_party/ascend/build/packaging.py"
    _load_packaging(source)._copy_ascend_tools(extdir, get_cmake_dir())


def _load_packaging(source):
    spec = importlib.util.spec_from_file_location("_flagtree_ascend_packaging", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Ascend packaging hook: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finalize_build_tree(build_lib):
    """Delegate packaging to the backend without importing its runtime."""
    source = Path(__file__).resolve().parents[3] / "third_party/ascend/build/packaging.py"
    _load_packaging(source).finalize_build_tree(build_lib)


def warn_stale_ascend_npu_ir_checkout(module):
    # DownloadManager reuses dst_path as-is when it already exists, so a
    # checkout left on a different commit would silently shadow the pin
    # registered for this build.
    if not module.commit_id or not os.path.isdir(os.path.join(module.dst_path, ".git")):
        return
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=module.dst_path, capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip().startswith(module.commit_id):
        return
    print(f"[WARNING] {module.dst_path} is at {head.stdout.strip()[:8]} but this build pins "
          f"AscendNPU-IR to {module.commit_id[:8]}; remove the directory (or run "
          f"'git checkout {module.commit_id}' inside it) before rebuilding")
