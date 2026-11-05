# Copyright 2026- FlagOS Contributors
# Copyright 2026- Xcoresigma Technology Co., Ltd

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

from triton.experimental.tle.raw.runtime import RawJITFunction

# This module must stay importable on non-Ascend builds: nothing here may
# import triton.language.extra.cann at module level. CANN-specific imports are
# deferred to CANNJITFunction construction / bitcode resolution time, which
# only happens on the Ascend backend.

# ccec aicore target per core type (Ascend 910B family), mirroring
# custom_ops/build_custom_ops.sh.
_ARCH_BY_CORE = {
    "VECTOR": "dav-c220-vec",
    "CUBE": "dav-c220-cube",
}

_DEFAULT_PIPELINE: Final[Dict[str, str]] = {
    "core": "vector",
    "pipe": "PIPE_V",
    "mode": "SIMD",
}

_BITCODE_FORMATS = ("bitcode", "bc")
_SOURCE_FORMATS = (None, "", "source", "cpp")

# Sibling BCs for cann source-JIT (e.g. custom_ops.mix.bc). Always late-linked
# via --link-aicore-bitcode; plain/mix are never llvm-linked into one module.
_op_extra_link_bcs: List[str] = []


def add_op_extra_link_bitcode(path: str) -> None:
    """Register an extra --link-aicore-bitcode path for a cann source-JIT op."""
    global _op_extra_link_bcs
    resolved = str(Path(path).resolve())
    if resolved not in _op_extra_link_bcs:
        _op_extra_link_bcs.append(resolved)


def get_op_extra_link_bitcodes() -> List[str]:
    """Op-local split BC siblings (plain is on the custom-op bitcode attr)."""
    return list(_op_extra_link_bcs)


@functools.lru_cache(maxsize=None)
def _find_ccec() -> str:
    candidates = []
    env = os.getenv("CCEC")
    if env:
        candidates.append(Path(env))
    which = shutil.which("ccec")
    if which:
        candidates.append(Path(which))
    npu_compiler = os.getenv("TRITON_NPU_COMPILER_PATH")
    if npu_compiler:
        candidates.append(Path(npu_compiler) / "ccec")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise RuntimeError("ccec not found: source the CANN environment first, or set $CCEC / $TRITON_NPU_COMPILER_PATH")


@functools.lru_cache(maxsize=None)
def _ccec_version(ccec: str) -> str:
    try:
        build = subprocess.run([ccec, "--version"], capture_output=True)
        return (build.stdout + build.stderr).decode(errors="replace")
    except OSError:
        return ""


def _template_include_dirs(extra: Sequence[Path] = ()) -> List[str]:
    dirs: List[Path] = []
    env = os.getenv("TLE_CANN_TEMPLATE_INCLUDE")
    if env:
        dirs.append(Path(env))
    else:
        import triton
        triton_root = Path(triton.__file__).resolve().parent  # .../triton
        candidates = [
            # Editable / in-tree: <repo>/python/triton -> <repo>/third_party/...
            triton_root.parent.parent / "third_party" / "ascend" / "AscendNPU-IR" / "bishengir" / "lib" / "Template" /
            "include",
            # Some installs keep AscendNPU-IR next to the package.
            triton_root / "backends" / "ascend" / "AscendNPU-IR" / "bishengir" / "lib" / "Template" / "include",
        ]
        # Source checkout used by this workspace when triton is a site-packages install.
        flash = os.getenv("FLAGTREE_ROOT")
        if flash:
            candidates.append(
                Path(flash) / "third_party" / "ascend" / "AscendNPU-IR" / "bishengir" / "lib" / "Template" / "include")
        for default in candidates:
            if default.is_dir():
                dirs.append(default)
                break
    dirs.extend(Path(p) for p in extra)
    return [str(d.resolve()) for d in dirs if d.is_dir()]


def _compile_bitcode(src: Path, arch: str, includes: Sequence[str]) -> str:
    """JIT-compile AscendC to plain + mix bitcode; late-link both (never merge).

    Returns the plain .bc path (custom-op ``bitcode`` attr). The mix sibling is
    registered via ``add_op_extra_link_bitcode`` for ``--link-aicore-bitcode``.
    """
    from triton.runtime.cache import get_cache_manager

    ccec = _find_ccec()
    key_payload = "\x00".join([
        "cann-jit",
        "split-plain-mix",
        hashlib.sha256(src.read_bytes()).hexdigest(),
        arch,
        ccec,
        _ccec_version(ccec),
        *includes,
    ])
    key = hashlib.sha256(key_payload.encode()).hexdigest()
    cache = get_cache_manager(key)
    cached_plain = cache.get_file("custom_ops.bc")
    cached_mix = cache.get_file("custom_ops.mix.bc")
    if cached_plain is not None and cached_mix is not None:
        add_op_extra_link_bitcode(cached_mix)
        return cached_plain

    common = [
        "-O2",
        "-x",
        "cce",
        "--cce-auto-sync=off",
        "--cce-aicore-only",
        "--cce-generic-addrspace=off",
        "-mllvm",
        "-disable-llvm-optzns",
        f"--cce-aicore-arch={arch}",
        "--cce-enable-print",
        "--cce-enable-sanitizer",
        "-std=c++17",
    ]
    for include in includes:
        common += ["-I", str(include)]

    with tempfile.TemporaryDirectory(prefix="tle-cann-jit-") as tmp:
        plain_bc = Path(tmp) / "op.bc"
        mix_bc = Path(tmp) / "op.mix.bc"
        variants = [([], plain_bc), (["-cce-enable-mix", "-mllvm", "-enable-mix=true"], mix_bc)]
        for extra_args, out_bc in variants:
            cmd = [ccec, *common, *extra_args, str(src), "-emit-llvm", "-c", "-o", str(out_bc)]
            build = subprocess.run(cmd, capture_output=True)
            if build.returncode != 0:
                raise RuntimeError("ccec failed\n"
                                   f"cmd: {' '.join(cmd)}\n"
                                   f"stderr:  {build.stderr.decode(errors='replace')}")
        plain_path = cache.put(plain_bc.read_bytes(), "custom_ops.bc", binary=True)
        mix_path = cache.put(mix_bc.read_bytes(), "custom_ops.mix.bc", binary=True)
        add_op_extra_link_bitcode(mix_path)
        return plain_path


class CANNJITFunction(RawJITFunction):
    """CANN custom op bound by @dialect(name="cann", ...).

    The instance duck-types the custom-op attribute interface consumed by
    triton.language.extra.cann.extension.custom_op (core / pipe / mode /
    symbol / bitcode), so tle_raw.call can lower it to hivm.custom without
    the string-keyed registry of the OP extension framework:

        @dialect(name="cann", format="bitcode",
                 file=Path(__file__).parent / "sort_topk.bc",
                 extern_func_name="custom_sort_1d_topk_proposals_float",
                 pipeline={"core": "vector", "pipe": "PIPE_V"})
        def sort_topk(*args, **kwargs):
            ...

        @triton.jit
        def kernel(src, tmp, out, K: tl.constexpr):
            out = tle_raw.call(sort_topk, [src, tmp, out, K], output_indices=[2])

    When format is "bitcode" / "bc" the file is referenced as-is. When format
    is omitted the file is treated as AscendC source and JIT-compiled to
    bitcode with ccec on first use; the produced bitcode is cached under the
    Triton cache directory (~/.triton/cache).

    Pass library=\"cannshmem\" for official cann/shmem headers (aclshmem_*
    inlined into the op BC).
    Pass library=\"hccl\" for AscendC device HCCL headers (Hccl::AllReduce /
    Commit; header-inline, no separate device bc).
    """

    def __init__(self, fn: Any, file, format: Optional[str] = None, extern_func_name: str = "",
                 pipeline: Optional[Dict[str, str]] = None, arch: Optional[str] = None, includes: Sequence = (),
                 extra_buffers=None, **kwargs) -> None:
        kwargs.setdefault("extern_func_name", extern_func_name)
        super().__init__(fn, **kwargs)
        self.format: Final[Optional[str]] = format
        self.symbol: Final[str] = extern_func_name or fn.__name__
        self.name: Final[str] = fn.__name__
        self.arch: Final[Optional[str]] = arch
        self._extra_includes: Final[Tuple[Path, ...]] = tuple(Path(p) for p in includes)
        if extra_buffers is not None:
            self.extra_buffers = extra_buffers

        if self.library == "cannshmem":
            from triton.experimental.tle.raw.cann.cannshmem.utils import enable_cannshmem
            enable_cannshmem(True)
        elif self.library == "hccl":
            from triton.experimental.tle.raw.cann.hccl.utils import enable_hccl
            enable_hccl(True)
        elif self.library:
            raise ValueError(f"unsupported @dialect(name='cann') library: {self.library!r} "
                             f"(expected '', 'cannshmem', or 'hccl')")

        if self.format not in _BITCODE_FORMATS and self.format not in _SOURCE_FORMATS:
            raise ValueError(f"unsupported @dialect(name='cann') format: {self.format!r} "
                             f"(expected 'bitcode'/'bc' or a source format)")

        merged = dict(_DEFAULT_PIPELINE)
        if pipeline:
            merged.update(pipeline)

        import triton.language.extra.cann.extension as al
        try:
            self.core = al.CORE[merged["core"].upper()]
            self.pipe = al.PIPE[merged["pipe"].upper()]
            self.mode = al.MODE[merged["mode"].upper()]
        except KeyError as e:
            raise ValueError(f"invalid pipeline metadata {pipeline!r}: unknown entry {e}") from None

        path = Path(file)
        if not path.is_absolute():
            # Relative paths resolve against the decorated function's source
            # directory first, so operators can ship source next to the call.
            try:
                base = Path(inspect.getabsfile(fn)).parent
                if (base / path).exists():
                    path = base / path
            except TypeError:
                pass
        self.file: Final[Path] = path
        self._bitcode: Optional[str] = None
        # Override the generic text-based TLE raw source cache key (see
        # triton.experimental.tle.raw.cache_key): this dialect's file may be
        # bitcode (binary) or AscendC source, so hash raw bytes instead. The
        # generic bind honors a pre-set attribute and leaves this one in place.
        self.__triton_tle_raw_source_cache_key__ = self._source_cache_key

    def _source_cache_key(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(self.file.resolve()).encode())
        hasher.update(self.file.read_bytes())
        return hasher.hexdigest()

    def _resolve_bitcode(self) -> str:
        if self.format in _BITCODE_FORMATS:
            resolved = str(self.file.resolve())
            assert self.file.exists(), f"@dialect(name='cann') bitcode file not found: {resolved}"
            return resolved
        if not self.file.exists():
            raise FileNotFoundError(f"@dialect(name='cann') source file not found: {self.file}")
        arch = self.arch
        if arch is None:
            arch = _ARCH_BY_CORE.get(self.core.name)
            if arch is None:
                raise ValueError(f"cannot infer ccec aicore arch for core={self.core.name!r}; "
                                 "pass arch=... explicitly")
        includes = _template_include_dirs(self._extra_includes)
        if self.library == "cannshmem":
            from triton.experimental.tle.raw.cann.cannshmem.utils import get_cannshmem_include_dirs
            includes = list(includes) + get_cannshmem_include_dirs()
        elif self.library == "hccl":
            from triton.experimental.tle.raw.cann.hccl.utils import get_hccl_include_dirs
            includes = list(includes) + get_hccl_include_dirs()
        return _compile_bitcode(self.file, arch, includes)

    @property
    def bitcode(self) -> str:
        if self._bitcode is None:
            self._bitcode = self._resolve_bitcode()
        return self._bitcode

    def __deepcopy__(self, memo: Dict[int, Any]) -> "CANNJITFunction":
        return self
