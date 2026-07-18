# Copyright 2026 FlagOS Contributors
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

import subprocess
import os
from pathlib import Path
import shutil

global registrar


def printinfo(msgs):
    print(f" [TLE-DIST-INFO]: {msgs}.")


class FlagCXRegistrar:

    def __init__(self, external):
        self.bitcode_name = "libflagcx_device.bc"
        self.shared_lib_name = "libflagcx.so"
        self._set_path(external)

    def _is_flagcx_recompile_required(self):
        ENVS = ("FLAGCX_RECOMPILE", "DIST_RECOMPILE")
        for env in ENVS:
            env_value = os.environ.get(env, "0")
            if env_value in ("1", "true", "True", "ON"):
                printinfo(f"Recompiling FlagCX due to {env}={env_value}\n ")
                return env_value
        return False

    def _is_cache_available(self):
        ENVS = ("FLAGCX_CACHE", "DIST_CACHE")

        for env in ENVS:
            env_value = os.environ.get(env, "1")
            if env_value in ("0", "false", "False", "OFF", "clean"):
                printinfo(f"Skipping using cache at {self.cache_lib_dir} due to {env}={env_value}\n ")
            if env_value in ("clean"):
                shutil.rmtree(self.cache_lib_dir, ignore_errors=True)
                os.makedirs(self.cache_lib_dir, exist_ok=True)
                printinfo(f"Cache {self.cache_lib_dir} cleaned due to {env}={env_value}\n ")
            return False
        return True

    def _set_path(self, external):
        submodule = external['backend']
        flagtree_cache = external['cache']
        flagtree_config = external['configs']
        backend_name = flagtree_config.flagtree_backend
        self.backend_name = "nvidia" if not backend_name else backend_name
        self.flagcx_src_dir = submodule.dst_path
        self.flagtree_dir = flagtree_config.flagtree_root_dir
        self.src_lib_dir = Path(self.flagcx_src_dir) / "build" / "lib"
        self.cache_lib_dir = Path(flagtree_cache.dir_path) / "flagcx"
        flagtree_cache._create_subdir(subdir_name="flagcx")
        for lib_name in (self.bitcode_name, self.shared_lib_name):
            src_path = self.src_lib_dir / lib_name
            cache_path = self.cache_lib_dir / lib_name
            setattr(self, f"{lib_name.split('.')[0]}_src_path", src_path)
            setattr(self, f"{lib_name.split('.')[0]}_cache_path", cache_path)

    def _get_runtime_path(self, lib_name):
        return {
            self.shared_lib_name:
            Path(self.flagtree_dir) / "python" / "triton" / "_C" / self.shared_lib_name, self.bitcode_name:
            Path(self.flagtree_dir) / "third_party" / self.backend_name / "backend" / "lib" / self.bitcode_name
        }.get(lib_name)

    def get_compile_cmds(self):
        nproc = os.cpu_count()
        return {
            self.bitcode_name: ["make", "-C", "bindings/ir/nvidia"], self.shared_lib_name: ["make", "-j",
                                                                                            str(nproc)]
        }

    def _compile_and_cache(self):
        cmds = self.get_compile_cmds()
        is_unused_cache = self._is_cache_available()
        is_recompile = self._is_flagcx_recompile_required()
        is_unused_cache = is_unused_cache or is_recompile

        for lib_name, cmd in cmds.items():
            cache_path = getattr(self, f"{lib_name.split('.')[0]}_cache_path")
            src_path = getattr(self, f"{lib_name.split('.')[0]}_src_path")
            runtime_path = self._get_runtime_path(lib_name)
            if cache_path.exists() and not is_unused_cache:
                printinfo(f"{lib_name} already exists in cache, skipping compilation ...")
                shutil.copy(cache_path, runtime_path)
            elif src_path.exists() and not is_recompile:
                printinfo(f"{lib_name} already exists in build directory, copying to cache...")
                shutil.copy(src_path, cache_path)
                shutil.copy(src_path, runtime_path)
                printinfo(f"{lib_name} copied from {src_path} to cache at {cache_path}")
                printinfo(f"{lib_name} copied from {src_path} to cache at {runtime_path}")
            else:
                printinfo(f"Compiling {lib_name} in {self.flagcx_src_dir}...")
                subprocess.run(cmd, cwd=self.flagcx_src_dir, check=True)
                if not src_path.exists():
                    raise FileNotFoundError(f"Expected {lib_name} not found: {src_path}")
                printinfo(f"{lib_name} compilation completed.")
                shutil.copy(src_path, cache_path)
                shutil.copy(src_path, runtime_path)
                printinfo(f"{lib_name} copied from {src_path} to cache at {cache_path}")
                printinfo(f"{lib_name} copied from {src_path} to cache at {runtime_path}")

    def _copy_required_files(self):
        dst = Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "flagcx_wrapper.py"
        src = Path(self.flagcx_src_dir) / "plugin" / "interservice" / "flagcx_wrapper.py"
        shutil.copy(src, dst)
        printinfo(f"flagcx_wrapper.py copied from {src} to {dst}")
        dst = Path(self.flagtree_dir) / "third_party" / "nvidia" / "backend" / "flagcx_wrapper.py"
        shutil.copy(src, dst)
        printinfo(f"flagcx_wrapper.py copied from {src} to {dst}")
        dst = Path(self.flagtree_dir) / "python" / "triton" / "experimental" / "tle" / "language" / "include"
        src = Path(self.flagcx_src_dir) / "flagcx" / "include"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        printinfo(f"FlagCX headers copied from {src} to {dst}")

    def run(self):
        self._compile_and_cache()
        self._copy_required_files()


def handle_flagcx(*args, **kwargs):
    global registrar
    registrar = FlagCXRegistrar(kwargs)
    registrar.run()
