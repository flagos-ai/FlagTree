import os
import runpy  # FlagPrism: load the external build policy.
import shutil
from pathlib import Path
from .utils.tools import flagtree_configs as configs
from .utils.tools import download_flagtree_third_party


# FlagPrism: resolve its dependency through FlagTree's existing package helpers.
def get_flagprism_dependency_cmake_args(_build_ext, get_thirdparty_packages, get_json_package_info):
    # FlagPrism: reuse the preloaded nlohmann/json tree in offline builds.
    if not os.getenv("JSON_SYSPATH", "").strip():
        user_home = os.getenv("TRITON_HOME") or os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH")
        cache_root = Path(user_home or Path.home()) / ".triton"
        json_path = cache_root / "json"
        if (json_path / "include" / "nlohmann" / "json.hpp").is_file():
            os.environ["JSON_SYSPATH"] = str(json_path)
    return get_thirdparty_packages([get_json_package_info()])


class FlagPrismSetup:
    """FlagPrism: manage optional component build and package integration."""

    def __init__(self, project_root, dependency_cmake_args):
        # FlagPrism: use one source-root base regardless of the caller's cwd.
        self.project_root = Path(project_root).resolve()
        backend = configs.flagtree_backend or ""
        # FlagPrism: register all supported integration backends together.
        supported_backends = {"ascend", "iluvatar", "mthreads"}
        default = "ON" if backend in supported_backends else "OFF"
        self.enabled = self._check_env_flag("TRITON_BUILD_FLAGPRISM", default)
        self.build_config = None
        self._dependency_cmake_args = dependency_cmake_args

        if self.enabled and backend not in supported_backends:
            # FlagPrism: report the newly supported mthreads backend.
            raise RuntimeError("TRITON_BUILD_FLAGPRISM is only supported when "
                               "FLAGTREE_BACKEND=ascend, iluvatar, or mthreads.")
        if not self.enabled:
            return
        if self._check_env_flag("TRITON_BUILD_PROTON"):
            raise RuntimeError("TRITON_BUILD_FLAGPRISM and TRITON_BUILD_PROTON cannot both be enabled. "
                               "Set one of them to OFF.")

        # FlagPrism replaces Proton for the supported backend builds.
        os.environ["TRITON_BUILD_PROTON"] = "OFF"
        # FlagPrism: resolve external checkouts relative to the project root.
        source_override = os.environ.get("FLAGPRISM_SOURCE_DIR", "").strip()
        source_root = Path(source_override) if source_override else Path("third_party") / "FlagPrism"
        if not source_root.is_absolute():
            source_root = self.project_root / source_root
        source_root = source_root.resolve()
        # FlagPrism: never download a different checkout for an invalid override.
        if source_override and not source_root.is_dir():
            raise RuntimeError(f"FLAGPRISM_SOURCE_DIR must point to an existing directory: {source_root}")
        # Keep FlagPrism as an external checkout. A local directory or symlink
        # is authoritative; only bootstrap the registered dependency when it
        # is absent.
        if not source_root.exists():
            download_flagtree_third_party("FlagPrism", condition=True, required=True)

        helper_path = source_root / "python" / "flagprism_build.py"
        if not helper_path.is_file():
            # FlagPrism: identify incomplete overrides instead of suggesting a download.
            if source_override:
                raise RuntimeError(f"FLAGPRISM_SOURCE_DIR does not contain python/flagprism_build.py: {source_root}")
            raise RuntimeError("FlagPrism sources are missing. Run the Python package build "
                               "to download third-party dependencies.")
        policy = runpy.run_path(str(helper_path), run_name="_flagprism_build")
        # FlagPrism: keep CMake and setuptools on the same external source tree.
        self.build_config = policy["create_build_config"](self.project_root, source_root)

        legacy_link = self.project_root / "python" / "triton" / "profiler"
        if legacy_link.is_symlink():
            legacy_link.unlink()

    @staticmethod
    def _check_env_flag(name: str, default: str = "") -> bool:
        return os.getenv(name, default).upper() in ("ON", "1", "YES", "TRUE", "Y")

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)

    def _remove_legacy_gateway(self, build_lib: str) -> None:
        triton_root = Path(build_lib) / "triton"
        self._remove_path(triton_root / "_flagprism.py")
        for artifact in (triton_root / "__pycache__").glob("_flagprism.*.pyc"):
            self._remove_path(artifact)

    def cmake_args(self, build_lib: str) -> list[str]:
        if self.build_config is None:
            return ["-DTRITON_BUILD_FLAGPRISM=OFF"]
        return self.build_config.cmake_args(build_lib)

    def dependency_cmake_args(self, build_ext) -> list[str]:
        if not self.enabled:
            return []
        return self._dependency_cmake_args(build_ext)

    def prepare_build_tree(self, build_lib: str) -> None:
        # The gateway now belongs to flagtree; reused build trees must not
        # repackage the former triton._flagprism module.
        self._remove_legacy_gateway(build_lib)
        if self.build_config is not None:
            self.build_config.prepare_build_tree(build_lib)
            return
        build_root = Path(build_lib) / "flagtree"
        self._remove_path(build_root / "debugger")
        self._remove_path(build_root / "profiler")

    def finalize_build_tree(self, build_lib: str) -> None:
        if self.build_config is not None:
            self.build_config.finalize_build_tree(build_lib)
        else:
            self.prepare_build_tree(build_lib)
        self._remove_legacy_gateway(build_lib)

    def packages(self) -> tuple[str, ...]:
        if self.build_config is None:
            return ()
        return self.build_config.packages()

    def package_dirs(self) -> tuple[tuple[str, str], ...]:
        if self.build_config is None:
            return ()
        return self.build_config.package_dirs()

    def console_scripts(self) -> list[str]:
        if self.build_config is None:
            return []
        return self.build_config.console_scripts()
