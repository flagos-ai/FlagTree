# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil
import subprocess
from pathlib import Path


def printinfo(message):
    print(f" [TLE-DIST-INFO]: {message}.")


class FlagCXRegistrar:

    def __init__(self, external):
        self.bitcode_name = "libflagcx_device.bc"
        self.shared_lib_name = "libflagcx.so"
        self._set_path(external)

    def _is_flagcx_recompile_required(self):
        for env in ("FLAGCX_RECOMPILE", "DIST_RECOMPILE"):
            value = os.environ.get(env, "0")
            if value in ("1", "true", "True", "ON"):
                printinfo(f"Recompiling FlagCX due to {env}={value}\n ")
                return True
        return False

    def _is_cache_available(self):
        for env in ("FLAGCX_CACHE", "DIST_CACHE"):
            value = os.environ.get(env, "1")
            if value in ("0", "false", "False", "OFF", "clean"):
                printinfo(f"Skipping cache at {self.cache_lib_dir} due to {env}={value}\n ")
                if value == "clean":
                    shutil.rmtree(self.cache_lib_dir, ignore_errors=True)
                    os.makedirs(self.cache_lib_dir, exist_ok=True)
                return False
        return True

    def _set_path(self, external):
        submodule = external["backend"]
        cache = external["cache"]
        configs = external["configs"]
        self.backend_name = configs.flagtree_backend or "nvidia"
        self.flagcx_src_dir = submodule.dst_path
        self.flagtree_dir = configs.flagtree_root_dir
        self.src_lib_dir = Path(self.flagcx_src_dir) / "build" / "lib"
        self.cache_lib_dir = Path(cache.dir_path) / "flagcx"
        cache._create_subdir(subdir_name="flagcx")
        for name in (self.bitcode_name, self.shared_lib_name):
            setattr(self, f"{name.split('.')[0]}_src_path", self.src_lib_dir / name)
            setattr(self, f"{name.split('.')[0]}_cache_path", self.cache_lib_dir / name)

    def _get_runtime_path(self, name):
        if name == self.shared_lib_name:
            return Path(self.flagtree_dir) / "python" / "triton" / "_C" / name
        return Path(self.flagtree_dir) / "third_party" / self.backend_name / "backend" / "lib" / name

    def get_compile_cmds(self):
        return {
            self.bitcode_name: ["make", "-C", "bindings/ir/nvidia"],
            self.shared_lib_name: ["make", "-j", str(os.cpu_count())],
        }

    def _compile_and_cache(self):
        use_cache = self._is_cache_available()
        recompile = self._is_flagcx_recompile_required()
        for name, command in self.get_compile_cmds().items():
            cache_path = getattr(self, f"{name.split('.')[0]}_cache_path")
            src_path = getattr(self, f"{name.split('.')[0]}_src_path")
            runtime_path = self._get_runtime_path(name)
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists() and use_cache and not recompile:
                shutil.copy(cache_path, runtime_path)
            elif src_path.exists() and not recompile:
                shutil.copy(src_path, cache_path)
                shutil.copy(src_path, runtime_path)
            else:
                subprocess.run(command, cwd=self.flagcx_src_dir, check=True)
                if not src_path.exists():
                    raise FileNotFoundError(f"Expected {name} not found: {src_path}")
                shutil.copy(src_path, cache_path)
                shutil.copy(src_path, runtime_path)

    def _copy_required_files(self):
        source = Path(self.flagcx_src_dir) / "plugin" / "interservice" / "flagcx_wrapper.py"
        destinations = [
            Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" /
            "flagcx_wrapper.py",
            Path(self.flagtree_dir) / "third_party" / "nvidia" / "backend" / "flagcx_wrapper.py",
        ]
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, destination)
        destination = Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "include"
        source = Path(self.flagcx_src_dir) / "flagcx" / "include"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination)

    def run(self):
        self._compile_and_cache()
        self._copy_required_files()


def handle_flagcx(*args, **kwargs):
    FlagCXRegistrar(kwargs).run()
