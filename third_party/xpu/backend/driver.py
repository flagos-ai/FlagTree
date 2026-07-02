import os
import sys
import hashlib
import shutil
import tempfile
import functools
from typing import List
from pathlib import Path

# from third_party.xcn.backend.driver import library_dirs
from triton.runtime.build import _build
from triton.runtime.cache import get_cache_manager
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver

dirname = os.path.dirname(os.path.realpath(__file__))


@functools.lru_cache(maxsize=None)
def get_xpu_library_dirs(arch: int = -1):
    # 因为这里是还没有进入到compile阶段的一个判定，然后xpu4及以上不走这个target，所以就暂时仍使用xpu3的头文件和链接库，之后会在compile阶段再次判定
    if arch == -1:
        include_dir = [os.path.join(dirname, "xpu3", "include")]
        libdevice_dir = os.path.join(dirname, "xpu3", "lib")
        library_dir = os.path.join(dirname, "xpu3", "so")
        libraries = ["xpurt"]
        return include_dir, [libdevice_dir, library_dir], libraries

    include_dir = [os.path.join(dirname, f"xpu{arch}", "include")]
    libdevice_dir = os.path.join(dirname, f"xpu{arch}", "lib")
    library_dir = os.path.join(dirname, f"xpu{arch}", "so")
    if os.path.exists(os.path.join(library_dir, "libLaunch_shared.a")) or os.path.exists(
            os.path.join(library_dir, "liblaunch_shared.so")):
        libraries = ["launch", "xpurt", "launch_shared"]
    else:
        libraries = ["launch", "xpurt"]
    return include_dir, [libdevice_dir, library_dir], libraries


def get_xpu_spec(xpu_arch, is_sdnn=False):
    """
    `is_sdnn=False`: return a tuple represents (num_clusters, num_cores)

    `is_sdnn=True`: return a tuple represents (num_sdnns, num_cores)
    """
    if xpu_arch == 2:
        return (8, 8) if is_sdnn else (8, 64)
    elif xpu_arch == 3:
        return (12, 8) if is_sdnn else (12, 64)
    elif xpu_arch == 4:
        return (6, 8) if is_sdnn else (12, 64)
    else:
        raise RuntimeError(f"Unknown XPU architecture: {xpu_arch}")


def compile_module_from_src(src, name, arch: int = -1):
    _print_startup_banner_once()
    # Ensure a GLIBCXX_3.4.30-capable libstdc++ is loaded before the launcher .so
    # is dlopened, so no manual LD_LIBRARY_PATH/LD_PRELOAD is needed at runtime.
    _preload_libstdcxx()
    # Fold the backend install dir into the cache key. The launcher .so bakes
    # this dir into its RUNPATH (see _xpu_runtime_rpath_flags), so a cached
    # launcher built under a different install path would point at a stale/
    # missing liblaunch_shared.so. Keying on `dirname` invalidates such caches
    # automatically when the package is moved or reinstalled elsewhere.
    key = hashlib.sha256((src + dirname).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    cache_path = cache.get_file(f"{name}.so")

    if cache_path is None:
        include_dir, library_dirs, libraries = get_xpu_library_dirs(arch)
        _build_debug = os.environ.get("TRITON_XPU_BUILD_DEBUG", "0") not in ("", "0", "false", "False")
        if _build_debug:
            _dump_build_inputs(name, arch, include_dir, library_dirs, libraries)
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "main.c")
            with open(src_path, "w") as f:
                f.write(src)
            # Always dump launcher source for post-mortem inspection (mirror Triton 3.0 behavior).
            with open("/tmp/triton36_launcher_main.c", "w") as _dumpf:
                _dumpf.write(src)
            if _build_debug:
                # Preserve source for post-mortem inspection.
                import shutil as _sh
                _sh.copy(src_path, f"/tmp/triton_xpu_{name}_main.c")
                print(f"[BUILD-DEBUG] saved src -> /tmp/triton_xpu_{name}_main.c", flush=True)
            try:
                # driver.c contains C++ constructs (static_cast, new, reinterpret_cast),
                # so force g++ as compiler. ccflags is appended after src in cc_cmd, so
                # `-x c++` there doesn't take effect; switching CC is the reliable fix.
                _saved_cc = os.environ.get("CC")
                _gxx = shutil.which("g++")
                if _gxx is not None:
                    os.environ["CC"] = _gxx
                try:
                    so = _build(name, src_path, tmpdir, library_dirs, include_dir, libraries,
                                _xpu_runtime_rpath_flags(library_dirs))
                finally:
                    if _saved_cc is None:
                        os.environ.pop("CC", None)
                    else:
                        os.environ["CC"] = _saved_cc
            except Exception as build_exc:
                _dump_build_failure(name, arch, src_path, library_dirs, include_dir, libraries, build_exc)
                raise
            if _build_debug:
                _dump_build_output(name, so, library_dirs)
            with open(so, "rb") as f:
                cache_path = cache.put(f.read(), f"{name}.so", binary=True)

    # check  compiled library cache
    if not os.path.exists(cache_path):
        raise RuntimeError(f"Compiled library not found: {cache_path}")

    if not os.access(cache_path, os.R_OK):
        raise RuntimeError(f"Compiled library not readable: {cache_path}")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, cache_path)
        if spec is None:
            raise RuntimeError(f"Could not create module spec from {cache_path}")
        # NOTE: For ExtensionFileLoader, module_from_spec() itself triggers
        # dlopen (via create_dynamic), so symbol-resolution failures surface
        # here, not in exec_module(). The except-block below MUST cover both.
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        return mod
    except Exception as e:
        try:
            _dump_loader_failure_debug(cache_path, library_dirs, e)
        except Exception as dump_exc:  # noqa: BLE001
            print(f"[debug] _dump_loader_failure_debug itself failed: {dump_exc}", flush=True)
        print(f"Error loading module {name} from {cache_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        raise


def _flatten_library_dirs(library_dirs):
    flat = []
    for d in library_dirs:
        if isinstance(d, (list, tuple)):
            flat.extend(d)
        else:
            flat.append(d)
    return flat


@functools.lru_cache(maxsize=1)
def _compatible_libstdcxx():
    """Locate a libstdc++.so.6 that exports GLIBCXX_3.4.30.

    The prebuilt liblaunch_shared.so requires GLIBCXX_3.4.30, which the system
    libstdc++ (e.g. Ubuntu 20.04 => 3.4.28) does not provide. The active Python
    environment (conda/venv) ships a newer libstdc++, so resolve it from the
    interpreter prefixes. Returns an absolute path or None.
    """
    import glob
    candidates = []
    for prefix in (sys.prefix, sys.base_prefix):
        libdir = os.path.join(prefix, "lib")
        candidates.append(os.path.join(libdir, "libstdc++.so.6"))
        candidates.extend(sorted(glob.glob(os.path.join(libdir, "libstdc++.so.6.*"))))
    seen = set()
    for cand in candidates:
        if cand in seen or not os.path.isfile(cand):
            continue
        seen.add(cand)
        try:
            with open(cand, "rb") as f:
                blob = f.read()
        except OSError:
            continue
        if b"GLIBCXX_3.4.30" in blob:
            return cand
    return None


_libstdcxx_preloaded = False


def _preload_libstdcxx():
    """Preload a GLIBCXX_3.4.30-capable libstdc++ into the global namespace.

    Must run before the launcher .so is dlopened so its (and liblaunch_shared.so's)
    NEEDED libstdc++.so.6 binds to this newer library instead of the system one.
    Idempotent and best-effort; failures are non-fatal.
    """
    global _libstdcxx_preloaded
    if _libstdcxx_preloaded:
        return
    path = _compatible_libstdcxx()
    if path is None:
        return
    try:
        import ctypes
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        _libstdcxx_preloaded = True
    except OSError:
        pass


def _xpu_runtime_rpath_flags(library_dirs):
    flags = []
    seen = set()
    dirs = list(_flatten_library_dirs(library_dirs))
    # Bake the env libstdc++ dir into the launcher RUNPATH so a cached launcher
    # resolves GLIBCXX_3.4.30 without any manual LD_LIBRARY_PATH/LD_PRELOAD.
    _libcxx = _compatible_libstdcxx()
    if _libcxx is not None:
        dirs.append(os.path.dirname(_libcxx))
    for libdir in dirs:
        if not libdir or libdir in seen:
            continue
        seen.add(libdir)
        if os.path.isdir(libdir):
            flags.append(f"-Wl,-rpath,{libdir}")
    return flags


def _md5_short(path, n=12):
    import hashlib as _h
    try:
        with open(path, "rb") as f:
            return _h.md5(f.read()).hexdigest()[:n]
    except Exception:  # noqa: BLE001
        return "?"


def _dump_build_inputs(name, arch, include_dir, library_dirs, libraries):
    """Print resolved build inputs before invoking _build (gated by TRITON_XPU_BUILD_DEBUG)."""
    print(f"\n========== TRITON-XPU BUILD-DEBUG: inputs for '{name}' arch={arch} ==========", flush=True)
    print(f"include_dir  = {include_dir}", flush=True)
    print(f"library_dirs = {library_dirs}", flush=True)
    print(f"libraries    = {libraries}", flush=True)
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '<unset>')}", flush=True)
    # show which physical .so each requested library will resolve to in the given dirs
    for d in _flatten_library_dirs(library_dirs):
        if not os.path.isdir(d):
            continue
        for lib in libraries:
            cand = os.path.join(d, f"lib{lib}.so")
            if os.path.exists(cand):
                print(f"  resolves: -l{lib} -> {cand}  md5={_md5_short(cand)}  size={os.path.getsize(cand)}",
                      flush=True)
    print("=" * 70, flush=True)


def _dump_build_output(name, so_path, library_dirs):
    """Print readelf/ldd of the freshly produced .so (gated by TRITON_XPU_BUILD_DEBUG)."""
    import subprocess
    print(f"\n========== TRITON-XPU BUILD-DEBUG: output {so_path}  md5={_md5_short(so_path)} ==========", flush=True)
    for cmd in (["ls", "-la", so_path], ["readelf", "-d", so_path], ["ldd", so_path]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.stdout:
                print(r.stdout, end="", flush=True)
            if r.stderr:
                print(r.stderr, end="", flush=True)
        except Exception as ee:  # noqa: BLE001
            print(f"[build-debug] {cmd!r} failed: {ee}", flush=True)
    print("=" * 70, flush=True)


def _dump_build_failure(name, arch, src_path, library_dirs, include_dir, libraries, exc):
    """Always-on dump when _build itself raises. Saves source for post-mortem."""
    import shutil
    print(f"\n========== TRITON-XPU BUILD-FAILED: '{name}' arch={arch} ==========", flush=True)
    print(f"exception: {type(exc).__name__}: {exc}", flush=True)
    print(f"include_dir  = {include_dir}", flush=True)
    print(f"library_dirs = {library_dirs}", flush=True)
    print(f"libraries    = {libraries}", flush=True)
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '<unset>')}", flush=True)
    # preserve the temp source so user can rerun the failing compile manually
    try:
        keep = os.path.join(tempfile.gettempdir(), f"triton_xpu_failed_{name}_{os.getpid()}.c")
        shutil.copyfile(src_path, keep)
        print(f"failing source copied to: {keep}", flush=True)
    except Exception as ee:  # noqa: BLE001
        print(f"[build-debug] could not preserve source: {ee}", flush=True)
    print("=" * 70, flush=True)


def _dump_loader_failure_debug(cache_path, library_dirs, exc):
    """Dump diagnostic info when the JIT-launcher .so fails to import.

    Designed to be CI-friendly: prints everything needed to pin down
    cross-machine ABI / RUNPATH / LD_LIBRARY_PATH issues without requiring
    the user to ssh into the failing host.
    """
    import subprocess
    import hashlib as _hashlib
    import re as _re

    def _section(title):
        print(f"\n========== TRITON-XPU DEBUG: {title} ==========", flush=True)

    def _run(cmd, **kw):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, **kw)
            if r.stdout:
                print(r.stdout, end="", flush=True)
            if r.stderr:
                print(r.stderr, end="", flush=True)
            return r
        except Exception as ee:  # noqa: BLE001
            print(f"[debug] command {cmd!r} failed: {ee}", flush=True)
            return None

    def _md5(path, n=12):
        try:
            with open(path, "rb") as f:
                return _hashlib.md5(f.read()).hexdigest()[:n]
        except Exception:  # noqa: BLE001
            return "?"

    _section(f"loader exception: {type(exc).__name__}: {exc}")

    # ---- 0. process / env basics ----
    _section("environment")
    print(f"sys.executable = {sys.executable}", flush=True)
    print(f"sys.version    = {sys.version.splitlines()[0]}", flush=True)
    for k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "TRITON_XPU_ARCH", "CUDA_VISIBLE_DEVICES"):
        print(f"{k}={os.environ.get(k, '<unset>')}", flush=True)

    # ---- 1. cache_path itself ----
    _section(f"target launcher: {cache_path}")
    _run(["ls", "-la", cache_path])
    _run(["readelf", "-d", cache_path])
    _run(["ldd", cache_path])

    # ---- 2. liblaunch_shared.so & sibling .so ----
    library_dir_flat = []
    for d in library_dirs:
        if isinstance(d, (list, tuple)):
            library_dir_flat.extend(d)
        else:
            library_dir_flat.append(d)
    for d in library_dir_flat:
        liblaunch = os.path.join(d, "liblaunch_shared.so")
        if not os.path.exists(liblaunch):
            continue
        _section(f"liblaunch_shared.so @ {liblaunch}  md5={_md5(liblaunch)}")
        _run(["readelf", "-d", liblaunch])
        # undefined xpurtc symbols (the failing symbol class)
        try:
            r = subprocess.run(
                ["nm", "-DC", "--undefined-only", liblaunch],
                capture_output=True,
                text=True,
                timeout=30,
            )
            undef = [ln for ln in r.stdout.splitlines() if "xpurtc" in ln]
            print(f"undefined xpurtc symbols ({len(undef)}):", flush=True)
            for ln in undef[:20]:
                print(f"  {ln}", flush=True)
        except Exception as ee:  # noqa: BLE001
            print(f"[debug] nm failed: {ee}", flush=True)
        # neighbouring libxpujitc.so
        sib = os.path.join(d, "libxpujitc.so")
        if os.path.exists(sib):
            _section(f"sibling libxpujitc.so @ {sib}  md5={_md5(sib)}")
            try:
                r = subprocess.run(
                    ["nm", "-DC", "--defined-only", sib],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                hits = [
                    ln for ln in r.stdout.splitlines() if _re.search(r"xpurtc::Kernel::(hash|code|size|xpu_arch)", ln)
                ]
                print(f"xpurtc::Kernel symbols defined here ({len(hits)}):", flush=True)
                for ln in hits[:10]:
                    print(f"  {ln}", flush=True)
            except Exception as ee:  # noqa: BLE001
                print(f"[debug] nm failed: {ee}", flush=True)

    # ---- 3. ALL libxpujitc.so visible to the loader ----
    _section("libxpujitc.so visible via LD_LIBRARY_PATH (and a few common roots)")
    seen = set()
    search_dirs = list(filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(":")))
    search_dirs.extend(library_dir_flat)
    # also typical roots known to contain stale copies in some envs
    for extra in [
            os.path.join(sys.prefix, "lib"),
            os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages",
                         "torch_xmlir"),
    ]:
        search_dirs.append(extra)
    for d in search_dirs:
        cand = os.path.join(d, "libxpujitc.so")
        if cand in seen or not os.path.isfile(cand):
            continue
        seen.add(cand)
        try:
            r = subprocess.run(
                ["nm", "-DC", "--defined-only", cand],
                capture_output=True,
                text=True,
                timeout=30,
            )
            has_hash = any("xpurtc::Kernel::hash() const" in ln for ln in r.stdout.splitlines())
        except Exception:  # noqa: BLE001
            has_hash = "?"
        size = os.path.getsize(cand)
        print(f"  hash_def={has_hash} size={size} md5={_md5(cand)} {cand}", flush=True)

    # ---- 4. LD_DEBUG=libs trace of the actual dlopen ----
    _section("LD_DEBUG=libs trace of dlopen(cache_path)")
    helper = ("import ctypes,sys; "
              f"ctypes.CDLL({cache_path!r}, ctypes.RTLD_GLOBAL)")
    env = os.environ.copy()
    env["LD_DEBUG"] = "libs"
    try:
        r = subprocess.run(
            [sys.executable, "-c", helper],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        # Filter to the interesting lines (libxpujitc / undefined / search path / RPATH)
        keep = []
        for ln in (r.stderr or "").splitlines():
            if ("libxpujitc" in ln or "undefined symbol" in ln or "RPATH" in ln or "RUNPATH" in ln
                    or "calling init: " in ln and "libxpujitc" in ln):
                keep.append(ln)
        print(f"(filtered LD_DEBUG output, {len(keep)} lines)", flush=True)
        for ln in keep[:80]:
            print(ln, flush=True)
    except Exception as ee:  # noqa: BLE001
        print(f"[debug] LD_DEBUG trace failed: {ee}", flush=True)

    # ---- 5. /proc/self/maps: which xpujitc/launch_shared are ALREADY mapped ----
    _section("/proc/self/maps (xpujitc / launch_shared / xpurt)")
    try:
        with open("/proc/self/maps") as f:
            seen_paths = set()
            for ln in f:
                if any(k in ln for k in ("libxpujitc", "liblaunch_shared", "libxpurt", "libxhpc")):
                    # only print the first occurrence per path (maps has many segments per file)
                    parts = ln.split()
                    p = parts[-1] if parts else ""
                    if p and p not in seen_paths:
                        seen_paths.add(p)
                        print(f"mapped: {p}  md5={_md5(p)}  size={os.path.getsize(p) if os.path.exists(p) else '?'}",
                              flush=True)
    except Exception as ee:  # noqa: BLE001
        print(f"[debug] /proc/self/maps read failed: {ee}", flush=True)

    # ---- 6. direct ctypes probe of sibling libxpujitc.so ----
    _section("direct ctypes.CDLL probe of sibling libxpujitc.so (RTLD_NOW)")
    for d in library_dir_flat:
        sib = os.path.join(d, "libxpujitc.so")
        if not os.path.exists(sib):
            continue
        probe = ("import os, ctypes\n"
                 f"try:\n"
                 f"    h = ctypes.CDLL({sib!r}, mode=os.RTLD_NOW)\n"
                 f"    print('probe-OK', {sib!r})\n"
                 f"except OSError as e:\n"
                 f"    print('probe-FAIL', {sib!r}, '->', e)\n")
        try:
            r = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if r.stdout:
                print(r.stdout, end="", flush=True)
            if r.stderr:
                print(r.stderr, end="", flush=True)
        except Exception as ee:  # noqa: BLE001
            print(f"[debug] probe failed: {ee}", flush=True)

    _section("END")


_STARTUP_BANNER_PRINTED = False


def _print_startup_banner_once():
    """Print backend dir contents once at first compile (gated by TRITON_XPU_BUILD_DEBUG).

    Helps confirm what wheel was actually installed on a CI machine before any
    failure happens.
    """
    global _STARTUP_BANNER_PRINTED
    if _STARTUP_BANNER_PRINTED:
        return
    _STARTUP_BANNER_PRINTED = True
    if os.environ.get("TRITON_XPU_BUILD_DEBUG", "0") in ("", "0", "false", "False"):
        return
    print("\n========== TRITON-XPU STARTUP BANNER ==========", flush=True)
    print(f"backend dir = {dirname}", flush=True)
    for arch_n in (3, 4):
        so_dir = os.path.join(dirname, f"xpu{arch_n}", "so")
        if not os.path.isdir(so_dir):
            continue
        print(f"  [xpu{arch_n}/so]", flush=True)
        for fn in sorted(os.listdir(so_dir)):
            if fn.endswith(".so") or fn.endswith(".a"):
                p = os.path.join(so_dir, fn)
                try:
                    print(f"    {fn}  size={os.path.getsize(p)}  md5={_md5_short(p)}", flush=True)
                except Exception:  # noqa: BLE001
                    pass
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '<unset>')}", flush=True)
    print("=" * 70, flush=True)


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == "*":
        return "void *"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def make_launcher(constants, signature, ids, metadata):
    xpu_arch = metadata.xpu_arch
    is_sdnn = metadata.is_sdnn
    ewtable = metadata.ewtable
    tensor_args = metadata.tensor_args
    kernel_name = metadata.name

    # Triton 3.6 keeps constexpr parameters in the signature dict; they are not
    # runtime arguments to the kernel ABI. The XPU LLVM kernel takes only the
    # non-constexpr params; gridX/gridY/gridZ are appended by the LoopGrid pass
    # and injected at device-launch time (see device/xpu3/launch.cpp), so they
    # must NOT be passed via the python launcher either.
    signature = {i: ty for i, ty in signature.items() if ty != "constexpr"}

    # Record the end of regular arguments;
    # subsequent arguments are architecture-specific descriptors, such as tensor descriptors for CUDA.
    arg_decls = ", ".join(f"{ty_to_cpp(ty)} arg{i}" + (f", int64_t arg{i}_numel" if ty[0] == "*" else "")
                          for i, ty in signature.items())

    def _extracted_type(ty):
        if ty[0] == "*":
            return "PyObject*"
        return ty_to_cpp(ty)

    def format_of(ty):
        return {
            "PyObject*": "O",
            "float": "f",
            "double": "d",
            "long": "l",
            "int8_t": "b",
            "int16_t": "h",
            "int32_t": "i",
            "int64_t": "l",  # TODO[dyq]: L?
            "uint8_t": "B",
            "uint16_t": "H",
            "uint32_t": "I",
            "uint64_t": "K",
        }[ty]

    def generate_kernel_params(signature, constants, type_mapping=None):
        params = []
        if type_mapping is None:
            type_mapping = {
                "*fp32": 1,
                "*f64": 2,
                "*int32": 3,
                "*int64": 4,
                "bool": 5,
                "*fp16": 6,
                "*bf16": 7,
                "int32": 8,
                "int64": 9,
                "int8": 10,
                "int16": 11,
            }

        for i, ty in signature.items():
            if i in constants:
                continue
            type_enum = type_mapping.get(ty, 0)
            params.append(f"{{{i}l, (int64_t)&arg{i}, (int64_t)(sizeof(arg{i})), "
                          f"{f'arg{i}_numel' if i in tensor_args else '0l'}, "
                          f"{type_enum}l}}")
        return f"{{ {', '.join(params) + (', ' if params else ' ')}{{0l, 0l, 0l, 0l, 0l}} }}"

    def generate_const_params(constants):
        params = []
        for i, val in constants.items():
            if isinstance(val, bool):
                params.append(f"{{{i}l, 5, {1 if val else 0}l}}")
            elif isinstance(val, int):
                params.append(f"{{{i}l, 4, {val}l}}")
        return f"{{ {', '.join(params) + (', ' if params else ' ')}{{0l, 0l, 0l}} }}"

    # drop type info
    args_format = "".join([format_of(_extracted_type(ty)) for ty in signature.values()])
    format = "iiiKKOOOO" + args_format

    args_list = ", " + ", ".join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ""

    def read_data_to_hexstr(file_name):
        if not file_name:
            return ""
        with open(file_name, "rb") as f:
            data = f.read()
            hex_lines = []
            for i in range(0, len(data), 128):
                chunk = data[i:i + 128]
                hex_string = ",".join(f"0x{byte:02x}" for byte in chunk)
                hex_lines.append(hex_string)
        return ",\n    ".join(hex_lines)

    # generate glue code
    src = f"""
#include <xpu/runtime.h>
#include <stdbool.h>
#include <Python.h>
#include <dlfcn.h>
#include <iostream>

// XPU_SPEC_START
static inline void xpuAssert(int code, const char *file, int line,
                             const char *call)
{{
   if (code != XPU_SUCCESS)
   {{
      const char* err_msg = xpu_strerror(code);
      char buf[1024] = {{0}};
      sprintf(buf, "%s:%d: %s -> %s(err_code: %d)",
              file, line, call, err_msg, code);
      PyGILState_STATE gil_state;
      gil_state = PyGILState_Ensure();
      PyErr_SetString(PyExc_RuntimeError, buf);
      PyGILState_Release(gil_state);
   }}
}}

#define XPU_CHECK(ans) {{ xpuAssert((ans), __FILE__, __LINE__, #ans); }}

enum {{
  kINVALID = 0,
  kL3,
  kGM
}};

static inline int xpu2PointerCheck(void *ptr) {{
  unsigned int ptr_high = (((unsigned long long) ptr) >> 32);
  unsigned int ptr_low = (((unsigned long long) ptr));
  if (ptr_high == 0 && ptr_low >= 0xC0000000 && ptr_low <= 0xC3FFFFFF) {{
      return kL3;
  }}
  if (ptr_high >= 8 && ptr_high <= 15) {{
      return kGM;
  }}
  printf("ptr_high = %u\\n", ptr_high);
  printf("ptr_low = %u\\n", ptr_low);
  return kINVALID;
}}

static inline int xpu3PointerCheck(void *ptr) {{
  // TODO: do it for XPU3.
  return kGM;
}}

static inline int xpu4PointerCheck(void *ptr) {{
  // TODO: do it for XPU4.
  return kGM;
}}

unsigned char ewtable[] = {{ {read_data_to_hexstr(ewtable)} }};

int xpuLaunchKernel(const char *kernel_name, void *func, int gridX, int gridY, int gridZ, int ncluster,
                    int ncores, void *stream, void **kernelParams, void **kernelConsts, void **extra);

static void _launch(int gridX, int gridY, int gridZ, int clusterDimX, int clusterDimY, int clusterDimZ, XPUStream stream, XPUFunc function{", " + arg_decls if len(arg_decls) > 0 else ""}) {{
  if (gridX*gridY*gridZ > 0) {{
    // printf("gridX: %d, gridY: %d, gridZ: %d\\n", gridX, gridY, gridZ);
    int nclusters = {get_xpu_spec(xpu_arch, is_sdnn)[0]};
    int ncores = {get_xpu_spec(xpu_arch, is_sdnn)[1]};
    int64_t kernel_params[][5] = {generate_kernel_params(signature, constants)};
    int64_t kernel_consts[][3] = {generate_const_params(constants)};
    void *extra_data[] = {{ {"&ewtable[0]" if ewtable else "nullptr"} }};

    // asm("int3");
    XPU_CHECK(xpuLaunchKernel(\"{kernel_name}\", function, gridX, gridY, gridZ, nclusters, ncores, stream, (void **)&kernel_params[0][0], (void **)&kernel_consts[0][0], &extra_data[0]));
  }}
}}
// XPU_SPEC_END

typedef struct _DevicePtrInfo {{
    void *dev_ptr;
    int64_t numel;
    bool valid;
}} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {{
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {{
    ptr_info.dev_ptr = PyLong_AsVoidPtr(obj);
    return ptr_info;
  }}
  if (obj == Py_None) {{
    // valid nullptr
    return ptr_info;
  }}
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  PyObject *len = PyObject_GetAttrString(obj, "numel");
  if(ptr && len){{
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret_ptr = PyObject_Call(ptr, empty_tuple, NULL);
    PyObject *ret_len = PyObject_Call(len, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    Py_DECREF(len);
    if (!PyLong_Check(ret_ptr)) {{
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    if (!PyLong_Check(ret_len)) {{
      PyErr_SetString(PyExc_TypeError, "numel method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }}
    ptr_info.dev_ptr = PyLong_AsVoidPtr(ret_ptr);
    if(!ptr_info.dev_ptr)
      return ptr_info;
    void *dev_ptr = PyLong_AsVoidPtr(ret_ptr);
    int64_t numel = PyLong_AsLong(ret_len);
    if (xpu{xpu_arch}PointerCheck(dev_ptr) == kINVALID) {{
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
    }}
    ptr_info.dev_ptr = dev_ptr;
    ptr_info.numel = numel;
    Py_DECREF(ret_ptr);  // Thanks ChatGPT!
    Py_DECREF(ret_len);  // Thanks ChatGPT!
    return ptr_info;
  }}
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr and numel method");
  ptr_info.valid = false;
  return ptr_info;
}}

static PyObject* launch(PyObject* self, PyObject* args) {{
  int gridX, gridY, gridZ;
  uint64_t _stream;
  uint64_t _function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *kernel_metadata = NULL;
  PyObject *launch_metadata = NULL;
  {" ".join([f"{_extracted_type(ty)} _arg{i}; " for i, ty in signature.items()])}
  if(!PyArg_ParseTuple(args, \"{format}\", &gridX, &gridY, &gridZ, &_stream, &_function,
                                           &kernel_metadata, &launch_metadata,
                                           &launch_enter_hook, &launch_exit_hook {args_list})) {{
    return NULL;
  }}

  int numWarps, numCtas, shared;
  int clusterDimX, clusterDimY, clusterDimZ;
  if (!PyArg_ParseTuple(kernel_metadata, \"iiiiii\", &numWarps, &numCtas, &shared, &clusterDimX, &clusterDimY, &clusterDimZ)) {{
    PyErr_SetString(PyExc_TypeError, "kernel_metadata must be a tuple");
    return NULL;
  }}

  // extract launch metadata
  if (launch_enter_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }}

  // raise exception asap
  {"; ".join([f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;" if ty[0] == "*" else "" for i, ty in signature.items()])};
  Py_BEGIN_ALLOW_THREADS;
  _launch(gridX, gridY, gridZ, clusterDimX, clusterDimY, clusterDimZ, (XPUStream)_stream, (XPUFunc)_function{", " + ", ".join(f"ptr_info{i}.dev_ptr, ptr_info{i}.numel" if ty[0] == "*" else f"_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ""});
  Py_END_ALLOW_THREADS;
  if (PyErr_Occurred()) {{
    return NULL;
  }}

  if(launch_exit_hook != Py_None){{
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;

  }}

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef ModuleMethods[] = {{
  {{"launch", (PyCFunction)launch, METH_VARARGS | METH_KEYWORDS, "Entry point for all kernels with this signature"}},
  {{NULL, NULL, 0, NULL}} // sentinel
}};

static struct PyModuleDef ModuleDef = {{
  PyModuleDef_HEAD_INIT,
  \"__triton_launcher\",
  NULL, //documentation
  -1, //size
  ModuleMethods
}};


PyMODINIT_FUNC PyInit___triton_launcher_xpu(void) {{
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {{
    return NULL;
  }}
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}}

"""
    return src


class XPUUtils(object):
    procedure: List[int] = [0, 0, 0]

    def __init__(self):
        mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "xpu_utils")
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties


def _get_launcher_fn_name(src):
    """Robustly extract kernel name from ASTSource, handling JITFunction variants."""
    fn = getattr(src, 'fn', None)
    if fn is None:
        return 'unknown'
    return (getattr(fn, 'name', None) or getattr(fn, '__name__', None)
            or getattr(getattr(fn, 'fn', None), '__name__', None) or str(fn))


def _print_triton_launcher_info(fn_name, signature, constants, args):
    """Print kernel launch info when PRINT_TRITON_FUNC env var is set.
    args layout: grid_0, grid_1, grid_2, stream, function, packed_metadata,
                 launch_metadata, launch_enter_hook, launch_exit_hook, *kernel_args
    """
    print(f"[Triton XPU Run] kernel={fn_name}")
    print(f"  grid      = ({args[0]}, {args[1]}, {args[2]})")
    if signature:
        print(f"  signature = {signature}", flush=True)
    if constants:
        print(f"  constants = {constants}", flush=True)
    for i, v in enumerate(args[9:]):
        if hasattr(v, 'shape'):
            print(f"  arg[{i}]   shape={tuple(v.shape)}, dtype={v.dtype}", flush=True)
        elif isinstance(v, int):
            print(f"  arg[{i}]   int={v}", flush=True)


class XPULauncher(object):

    def __init__(self, src, metadata):
        self._fn_name = _get_launcher_fn_name(src)
        self._signature = dict(src.signature) if hasattr(src, 'signature') else {}
        self._constants = dict(src.constants) if hasattr(src, 'constants') else {}
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()

        def cst_key(i):
            if isinstance(i, str):
                return src.fn.arg_names.index(i)
            if isinstance(i, tuple):
                assert len(i) == 1, f"unexpected nested constant key: {i}"
                return i[0]
            return i

        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        src = make_launcher(constants, signature, ids, metadata)
        mod = compile_module_from_src(src, "__triton_launcher_xpu", metadata.xpu_arch)
        self.launch = mod.launch

    def __call__(self, *args):
        if int(os.environ.get("PRINT_TRITON_FUNC", 0)):
            _print_triton_launcher_info(self._fn_name, self._signature, self._constants, args)
        # args = (gridX, gridY, gridZ, stream, function, metadata,
        #         launch_metadata, enter_hook, exit_hook, *bound_args)
        # Drop constexpr entries from bound_args: kernel ABI only takes
        # non-constexpr runtime params; gridX/Y/Z are injected by
        # xpuLaunchKernel itself (LoopGrid pass appended them as kernel args).
        fixed = args[:9]
        kernel_args = args[9:]
        filtered = [arg for arg, ty in zip(kernel_args, self._signature.values()) if ty != "constexpr"]
        self.launch(*fixed, *filtered)


@functools.lru_cache(maxsize=1)
def _xpu_utils_singleton():
    return XPUUtils()


class XPUDriver(GPUDriver):

    def __init__(self):
        self.utils = _xpu_utils_singleton()
        self.xpu_utils = _xpu_utils_singleton()
        self.launcher_cls = XPULauncher
        self.launcher_cls_xpu = XPULauncher
        super().__init__()
        # ==================== FLAGTREE XPU SYNC MARK ====================
        # GPUDriver wires these methods to torch.cuda.*, which initializes the
        # NVIDIA CUDA driver.  The XPU3 smoke/runtime path must not require a
        # CUDA-capable NVIDIA driver just to identify the active XPU target.
        # ==================== FLAGTREE XPU SYNC MARK ====================
        self.get_current_device = self._get_current_xpu_device
        self.set_current_device = self._set_current_xpu_device

    @staticmethod
    def _get_current_xpu_device():
        return int(os.environ.get("TRITON_XPU_DEVICE", os.environ.get("XPU_VISIBLE_DEVICE", "0")))

    @staticmethod
    def _set_current_xpu_device(device):
        os.environ["TRITON_XPU_DEVICE"] = str(device)

    @staticmethod
    def is_active():
        if os.environ.get("TRITON_BACKEND") == "xpu" or os.environ.get("FLAGTREE_BACKEND") == "xpu":
            return True
        try:
            from triton._flagtree_backend import FLAGTREE_BACKEND
            return FLAGTREE_BACKEND == "xpu"
        except Exception:
            return False

    def get_current_target(self):
        device = self.get_current_device()
        arch = self.utils.get_device_properties(device)["device_model"]
        warp_size = 1  # we don't have warp
        return GPUTarget("xpu", arch, warp_size)

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_active_torch_device(self):
        import torch
        return torch.device("xpu", self.get_current_device())

    def get_device_interface(self):
        import torch
        # On this stack PyTorch itself is built without USE_XPU. torch_xmlir
        # injects an XPU runtime shim under torch.cuda.* (see
        # "SYMBOL_REWRITE torch success" at import time). Returning torch.xpu
        # would hit PyTorch's native unsupported stub and trigger
        # "Torch not compiled with XPU enabled" inside _lazy_init().
        return torch.cuda

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        # torch_xmlir maps the XPU runtime onto torch.cuda.*; allocating with
        # device='xpu' would hit PyTorch's native unsupported XPU path. Use
        # device='cuda' so we go through the torch_xmlir shim.
        import torch
        return torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    def clear_cache(self, cache):
        cache.zero_()
