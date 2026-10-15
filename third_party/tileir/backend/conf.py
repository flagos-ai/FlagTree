import os
import triton


class TileIREnvConf:

    @staticmethod
    def enable_approx():
        # Enable approximate calculation, trading off numerical precision for performance gains
        return os.getenv("TILEIR_ENABLE_APPROX", "0") == "1"

    @staticmethod
    def enable_ftz():
        # Enable flush denormal to zero, trading off numerical precision for performance gains
        return os.getenv("TILEIR_ENABLE_FTZ", "0") == "1"

    @staticmethod
    def get_tileiras_path():
        # Resolution order: explicit env override > bundled binary > system PATH.
        env_path = os.getenv("TRITON_TILEIRAS_PATH", None)
        if env_path is not None:
            return os.path.join(env_path, "tileiras")

        # Bundled binary downloaded at build time into the self-contained
        # tileir CUDA root (backends/nvidia/tileir_cuda/bin/tileiras), alongside
        # the bundled 13.4 ptxas + libnvvm/libdevice.
        bundled_path = os.path.join(
            os.path.dirname(triton.__file__),
            "backends",
            "nvidia",
            "tileir_cuda",
            "bin",
            "tileiras",
        )
        if os.path.isfile(bundled_path) and os.access(bundled_path, os.X_OK):
            return bundled_path

        # Fall back to system PATH.
        from shutil import which

        tileiras_path = which("tileiras")
        if tileiras_path is None:
            raise RuntimeError("tileiras not found: TRITON_TILEIRAS_PATH unset, no bundled "
                               "binary, and not in PATH")
        return tileiras_path

    @staticmethod
    def get_tileir_cuda_home():
        # CUDA_HOME used by the tileiras subprocess to find ptxas + libnvvm +
        # libdevice for SM100 codegen. DERIVED purely from the resolved tileiras
        # location (tileiras always lives in <CUDA_HOME>/bin/tileiras): take
        # dirname(dirname(tileiras)). We deliberately do NOT read the system
        # CUDA_HOME env var, so a stale/older system CUDA (e.g. 13.2) can never
        # shadow the bundled 13.4 toolchain. Resolution therefore follows
        # get_tileiras_path():
        #   TRITON_TILEIRAS_PATH set -> dirname(TRITON_TILEIRAS_PATH)
        #   bundled                 -> backends/nvidia/tileir_cuda
        #   PATH                    -> dirname(dirname(which("tileiras")))
        # Injected ONLY into the tileiras subprocess env (see compiler.py).
        tileiras = TileIREnvConf.get_tileiras_path()
        return os.path.dirname(os.path.dirname(tileiras))
