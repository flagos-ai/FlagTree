"""HCU FlagCX runtime discovery and device-library linking."""

import functools
import hashlib
import json
from pathlib import Path
from triton.experimental._flagcx_config import (
    CudaFlagCXBackendAdapter,
    Distributed as _Distributed,
    FlagcxRuntimeConfig as _FlagcxRuntimeConfig,
)


class FlagcxRuntimeConfig(_FlagcxRuntimeConfig):
    # Discover preinstalled FlagCX packages through the shared NVIDIA policy.
    # The shared artifact cache does not identify the HCU build configuration.
    use_cache = False

    def __init__(self):
        super().__init__(Path(__file__).parent)


flagcx_rt_conf = FlagcxRuntimeConfig()
backend_adapter = CudaFlagCXBackendAdapter()


class Distributed(_Distributed):

    def __init__(self):
        # HCU matches 'flagcxDev*' declarations using this key, and compile-only
        # linking does not require a host runtime installation.
        super().__init__(flagcx_rt_conf, "flagcx", require_runtime=False)


def link_flagcx(llvm_mod, extern_libs, llvm, hcu, arch):
    """Import FlagCX before scanning its newly introduced DTK dependencies."""
    path = dict(extern_libs or ()).get("flagcx")
    if path and hcu.need_extern_lib(llvm_mod, "flagcx"):
        # Validate the selected library (including explicit extern_libs), only
        # when the kernel actually references FlagCX. Ordinary kernels may use
        # a different GPU from the installed FlagCX build.
        manifest = Path(path).resolve().parent / "metadata.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text())
            if metadata.get("backend") != "hcu":
                raise RuntimeError(f"FlagCX bundle is not built for HCU: {manifest}")
            if arch.split(":")[0] != metadata["arch"]:
                raise RuntimeError(f"FlagCX bitcode targets {metadata['arch']}, but this kernel targets {arch}. "
                                   "Rebuild FlagTree with DEVICE_BITCODE_ARCH set to the kernel GPU architecture.")
        llvm.link_extern_libs(llvm_mod, [path])


@functools.lru_cache(maxsize=128)
def _file_hash(filename, identity):
    """Cache by file identity/version, so replacing bitcode invalidates it."""
    digest = hashlib.sha256()
    with open(filename, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extern_libs_hash(extern_libs):
    """Include bitcode contents in the kernel cache key, not just its path."""
    result = []
    for name, filename in sorted(extern_libs):
        path = Path(filename).resolve()
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        result.append((name, filename, _file_hash(str(path), identity)))
    return tuple(result)
