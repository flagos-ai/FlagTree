from __future__ import annotations
import os
from pathlib import Path
from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    active_drivers = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(active_drivers) != 1:
        raise RuntimeError(f"{len(active_drivers)} active drivers ({active_drivers}). There should only be one.")
    return active_drivers[0]()


class DriverConfig:

    def __init__(self) -> None:
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default


driver = DriverConfig()


# flagtree backend specialization
def spec(function_name: str, *args, **kwargs):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func(*args, **kwargs)
    return None


# flagtree backend func specialization
def spec_func(function_name: str):
    if hasattr(driver.active, "spec"):
        spec = driver.active.spec
        if hasattr(spec, function_name):
            func = getattr(spec, function_name)
            return func
    return None


def _get_active_backend_name() -> str | None:
    backend = os.environ.get("FLAGTREE_BACKEND")
    if backend:
        return backend

    try:
        active_driver = driver.active
    except Exception:
        return None

    module_name = active_driver.__class__.__module__
    parts = module_name.split(".")
    if "backends" in parts:
        idx = parts.index("backends")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def spec_path(path_list: list):
    if not path_list:
        return

    backend = _get_active_backend_name()
    if not backend:
        return

    current_path = Path(path_list[0]).resolve()
    current_path_str = current_path.as_posix()

    repo_root = None
    rel_path = None

    # Source-tree mode:
    #   /root/lmc/FlagTree/python/triton/compiler
    #   /root/lmc/FlagTree/python/triton/experimental/gluon
    source_marker = "/python/triton"
    source_idx = current_path_str.find(source_marker)
    if source_idx != -1:
        repo_root = Path(current_path_str[:source_idx]).resolve()
        triton_root = repo_root / "python" / "triton"
        try:
            rel_path = current_path.relative_to(triton_root)
        except ValueError:
            repo_root = None
            rel_path = None

    # Installed mode:
    #   /usr/local/lib/python3.10/dist-packages/triton/compiler
    #   /usr/local/lib/python3.10/dist-packages/triton/experimental/gluon
    # In this mode, use FLAGTREE_SOURCE_ROOT to locate backend specializations.
    if repo_root is None or rel_path is None:
        source_root = os.environ.get("FLAGTREE_SOURCE_ROOT")
        if not source_root:
            source_root_file = Path(__file__).resolve().parents[1] / "FLAGTREE_SOURCE_ROOT"
            if source_root_file.is_file():
                source_root = source_root_file.read_text().strip()
        if not source_root:
            return

        repo_root = Path(source_root).resolve()
        installed_marker = "/triton/"
        installed_idx = current_path_str.find(installed_marker)
        if installed_idx == -1:
            return
        rel_path = Path(current_path_str[installed_idx + len(installed_marker):])

    candidate_backend_spec = repo_root / "third_party" / backend / "backend" / "spec" / "triton" / rel_path
    candidate_backend_python = repo_root / "third_party" / backend / "python" / "triton" / rel_path

    # Prefer backend/spec first, then legacy third_party/<backend>/python.
    for candidate_path in (candidate_backend_spec, candidate_backend_python):
        if candidate_path.is_dir():
            candidate = str(candidate_path)
            if candidate not in path_list:
                path_list.insert(0, candidate)
            return
