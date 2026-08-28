import os
import shutil
import subprocess
import hashlib
import torch
import errno
import logging
from posixpath import dirname
from pathlib import Path

from triton.backends.tsingmicro.logger_config import setup_logger

logger = setup_logger("tsingmicro_launch")

_dump_dir_cache = None
dump_cmd_count = 0

def _no_dump_file():
    tritonDump = os.getenv("TRITON_DUMP", "0").lower() in ("1", "true", "yes")
    if tritonDump:
        return False
    return True

def _get_dump_env_path():
    path = os.getenv("TRITON_DUMP_PATH", "")
    if not path:
        return ""
    os.makedirs(path, exist_ok=True)
    return path

def get_dump_ir_dir():
    global _dump_dir_cache
    if _dump_dir_cache is not None:
        return _dump_dir_cache

    base_dir = _get_dump_env_path()
    if not base_dir:
        return ""

    index = 0
    while True:
        device_id = torch.txda.current_device()
        run_name = f"run_{index}"
        card_name = f"device_{device_id}"
        full_path = os.path.join(base_dir, run_name, card_name)
        
        try:
            os.makedirs(full_path, exist_ok=False)
            _dump_dir_cache = full_path
            return _dump_dir_cache
        except OSError as e:
            if e.errno == errno.EEXIST:
                index += 1
                continue
            else:
                logger.error(f"Failed to create directory {full_path}: {e}")
                raise

def get_dump_dir():
    if _no_dump_file():
        return ""
    return get_dump_ir_dir()

def dump_ir(files):
    path = get_dump_ir_dir()
    if not path:
        return
    try:
        for f in files:
            shutil.copy(f, os.path.join(path, os.path.basename(f)))
    except Exception as e:
        logger.warning(f"exception: {e}")

def dump_cmd(cmd: list, flag: str):
    path = get_dump_ir_dir()
    if not path:
        return
    global dump_cmd_count
    if dump_cmd_count == 0:
        open_type = 'w'
    else:
        open_type = 'a'
    file_path = os.path.join(path, "cmds.txt")
    str_cmd = ' '.join(map(str, cmd))
    dump_cmd = f"{flag}:{str_cmd}\n\n"

    # 将字符串写入指定文件
    with open(file_path, open_type, encoding='utf-8') as f:
        f.write(dump_cmd)
    dump_cmd_count += 1

def runLoweringCmd(srcFile : str, destFile : str, args : list):
    try:
        isAlwaysCompile = os.getenv("TRITON_ALWAYS_COMPILE", "0").lower() in ("1", "true", "yes")
        if isAlwaysCompile or not os.path.exists(destFile):
            if os.getenv("MLIR_ENABLE_DUMP", "0") == "1":
                subprocess.check_call(args, stderr=subprocess.STDOUT)
            else:
                subprocess.check_call(args, stdout=subprocess.DEVNULL)
        else:
            logger.debug(f"Skip lowering {destFile}")
    except subprocess.CalledProcessError as e:
        dump_ir([srcFile])
        dump_cmd(args, "failed command")

def is_use_profile() :
    return os.getenv("ENABLE_PROFILING", "0").lower() in ("1", "true", "yes")


def is_use_tsm_profiler():
    return os.getenv("TSM_PROFILER_EN", "0").lower() in ("1", "true", "yes")


def is_enable_kernel_file_cache():
    return os.getenv("ENABLE_KERNEL_FILE_CACHE", "1").lower() in ("1", "true", "yes")

def get_kernel_cache_size():
    if is_enable_kernel_file_cache():
        kernel_size_str = os.getenv("KERNEL_FILE_SIZE", "1024")
        try:
            kernel_size = int(kernel_size_str)
            return str(kernel_size)
        except ValueError:
            raise ValueError(f"Illegal input KERNEL_FILE_SIZE '{kernel_size}', need Integer number.")
    else:
        raise ValueError("Must set ENABLE_KERNEL_FILE_CACHE=1 first")

def dump_ir_if_needed(files):
    if _no_dump_file():
        return
    path = get_dump_dir()
    if not path:
        return
    try:
        for f in files:
            shutil.copy(f, os.path.join(path, os.path.basename(f)))
    except Exception as e:
        logger.warning(f"exception: {e}")

def dump_file_if_needed(src_file, dest_file_name):
    if _no_dump_file():
        return

    path = get_dump_dir()
    if not path:
        return
    try:
        shutil.copy(src_file, os.path.join(path, dest_file_name))
    except Exception as e:
        logger.warning(f"exception: {e}")

def dump_cmd_if_needed(cmd: list, flag: str):
    if _no_dump_file():
        return
    path = get_dump_dir()
    if not path:
        return
    global dump_cmd_count
    if dump_cmd_count == 0:
        open_type = 'w'
    else:
        open_type = 'a'
    file_path = os.path.join(path, "cmds.txt")
    str_cmd = ' '.join(map(str, cmd))
    dump_cmd = f"{flag}:{str_cmd}\n\n"

    # 将字符串写入指定文件
    with open(file_path, open_type, encoding='utf-8') as f:
        f.write(dump_cmd)
    dump_cmd_count += 1

TSINGMICRO_DIR=os.path.join(os.getenv("HOME"), ".tsingmicro")

def get_llvm_system_path() -> str:
    path = os.getenv("LLVM_SYSPATH", "")
    if path == "":
        # path = os.path.join(os.path.dirname(__file__), "llvm/")
        path = os.path.join(TSINGMICRO_DIR, "llvm-22")
        if not Path(path).is_dir():
            raise Exception(f"error: cannot find llvm in {path}.")
    return path

def get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        llvm_path = get_llvm_system_path()
        path = llvm_path + "/bin"
        if not Path(path).is_dir():
            raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)

def get_tsm_opt_path() -> str:
    return os.path.join(os.path.dirname(__file__), "bin", "tsingmicro-opt")

def get_tx8_deps_path(sub_name: str) -> str:
    path = os.getenv("TX8_DEPS_ROOT", "")
    if path == "":
        # path = os.path.join(os.path.dirname(__file__), "tx8_deps/")
        path = os.path.join(TSINGMICRO_DIR, "tx8_deps")
        if not Path(path).is_dir():
            raise Exception(f"error: cannot find tx8_deps in {path}.")

    return os.path.join(path, sub_name)

def get_kuiper_path(sub_name: str) -> str:
    kuiper_path="/usr/local/kuiper"
    return os.path.join(kuiper_path, sub_name)

def get_tx8_profiler_path() -> str:
    path = os.path.join(dirname(get_tsm_opt_path()), "tx-profiler")
    return path

def is_dump_args_profile() :
    logger.getEffectiveLevel()
    if logger.getEffectiveLevel() <= logging.DEBUG:
        return 1
    else:
        return 0

def is_debug() :
    debug_value = os.getenv("DEBUG", "OFF").strip().upper()
    return debug_value in ["ON", "TRUE", "1", "YES"]

def calculate_str_md5(string : str):
    str_hash = hashlib.md5(string).hexdigest()
    return str_hash

def calculate_file_md5(file_path):
    with open(file_path, 'rb') as f:
        file_bytes = f.read()
    return calculate_str_md5(file_bytes)


def get_customized_ir() -> list:
    env_value = os.getenv("CUSTOMIZED_IR")
    if env_value is None:
        return []
    return [item.strip() for item in env_value.split(',') if item.strip()]

def get_customized_ir_file(ir_name: str):
    base_dir = _get_dump_env_path()
    if not base_dir:
        logger.warning(f"The dump path has not been set, so the customized ir function cannot be used.")
        return ""

    dest_file = os.path.join(base_dir, ir_name)
    if not os.path.exists(dest_file):
        logger.warning(f"The ir file: \"{dest_file}\" does not exist.")
        return ""
    return dest_file
