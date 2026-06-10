import os, subprocess, logging, sys

from pathlib import Path
from functools import wraps
import hashlib

def _show_perf():
    return os.environ.get("PERF_LOG_PRINT", None) != None

def _cache_key(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def run_command(command):
    try:
        # Execute the command
        subprocess.check_call(command)
        print(f"Command '{' '.join(command)}' executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(
            "An error occurred while executing the command:" + " ".join(command)
        )
        logging.error(e.stderr)  # Print standard error
        sys.exit(1)

def compile_to_linalg(jit_func, *args, **kwargs):
    kwargs["debug"] = None
    kwargs["to_ttsharedir"] = True
    return jit_func.warmup(*args, grid=[1,], **kwargs)
    
    
def with_env_vars(**env_vars):
    """Decorator to run a function with specific environment variables."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Save original values
            original_values = {}
            for key, value in env_vars.items():
                original_values[key] = os.environ.get(key)
                os.environ[key] = value
                
            try:
                # Run the function with modified environment
                return func(*args, **kwargs)
            finally:
                # Restore original values
                for key, value in original_values.items():
                    if value is None:
                        if key in os.environ:
                            del os.environ[key]
                    else:
                        os.environ[key] = value
        return wrapper
    return decorator

def ttsharedir_compare(jit_func, *args, **kargs):
    ttshared_kernel = compile_to_linalg(jit_func, *args, **kargs)
    golden_ir_dir = os.environ.get("EVAS_GOLDEN_IR_DIR")
    if golden_ir_dir is None:
        raise EnvironmentError("Please set EVAS_GOLDEN_IR_DIR.")
    golden_ir_file = Path(os.path.join(golden_ir_dir, ttshared_kernel.name + ".mlir"))
    if golden_ir_file.exists() and golden_ir_file.is_file():
        if golden_ir_file.read_bytes() == ttshared_kernel.kernel:
            print(f"Compiled IR compared with {golden_ir_file} successfully.")
            return True
        raise AssertionError(f"compiled IR compared with {golden_ir_file} failed")
    raise LookupError(f"{golden_ir_file} is not found")
