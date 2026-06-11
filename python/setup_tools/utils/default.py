import subprocess
import os
from pathlib import Path
import shutil

global registrar


def printinfo(msgs):
    print(f" [TLE-DIST-INFO]: {msgs}. \n")


class FlagCXRegistrar:

    def __init__(self, external):
        self.bitcode_name = "libflagcx_device.bc"
        self.shared_lib_name = "libflagcx.so"
        self._set_path(external)

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

        for lib_name, cmd in cmds.items():
            cache_path = getattr(self, f"{lib_name.split('.')[0]}_cache_path")
            src_path = getattr(self, f"{lib_name.split('.')[0]}_src_path")
            runtime_path = self._get_runtime_path(lib_name)
            if cache_path.exists():
                printinfo(f"{lib_name} already exists in cache, skipping compilation ...")
                shutil.copy(cache_path, runtime_path)
            elif src_path.exists():
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
                printinfo(f"[32m{lib_name} copied from {src_path} to cache at {runtime_path}")

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
