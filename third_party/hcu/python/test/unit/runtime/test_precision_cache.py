import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import triton


@pytest.mark.parametrize("source", ["core", "semantic", "standard", "driver", "precision"])
def test_hcu_provider_source_invalidates_cache(source, tmp_path):
    package = Path(triton.__file__).parent
    runtime = tmp_path / "runtime/triton"
    shutil.copytree(package, runtime, ignore=shutil.ignore_patterns("_C", "__pycache__"), symlinks=True)
    (runtime / "_C").symlink_to(package / "_C", target_is_directory=True)
    if source == "precision":
        provider = runtime / "backends/hcu/precision.py"
        old = "(.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)"
        new = "(.25, 1., 1.25, 1.75, 2.5, 3.5, 5.)"
    elif source == "driver":
        provider = runtime / "runtime/driver.py"
        old = "def reset_active(self) -> None:\n        self._active = self.default"
        new = old.replace("self.default", "None")
    else:
        provider = runtime / "language" / (source + ".py")
        if source == "core":
            old = "return _semantic.full(shape, value, dtype)"
            new = "return _semantic.full(shape, value + 1, dtype)"
        elif source == "semantic":
            old = "return self.splat(self.make_scalar(value, dtype), shape)"
            new = "return self.splat(self.make_scalar(value + 1, dtype), shape)"
        else:
            old = "return (x + (div - 1)) // div"
            new = old + " + 1"
    assert provider.resolve().is_relative_to(runtime.resolve())
    original = provider.read_text()
    assert original.count(old) == 1
    probe = tmp_path / "probe.py"
    probe.write_text('''import hashlib, json
from pathlib import Path
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler_hcu import HIPBackend
from triton.compiler import ASTSource
assert Path(triton.__file__).is_relative_to(Path(__file__).parent / "runtime")
@triton.jit
def encode(Y):
    packed = tl.ext.fp_to_fp4(tl.full((16,), .75, tl.float32))
    tl.store(Y + tl.arange(0, 8) + tl.cdiv(0, 1), packed)
target = GPUTarget("hip", "gfx936", 64)
kernel = triton.compile(ASTSource(encode, {"Y": "*u8"}), target=target)
print(json.dumps([HIPBackend(target).hash(), kernel.hash, hashlib.sha256(kernel.asm["ttir"].encode()).hexdigest()]))
''')
    env = dict(os.environ, PYTHONPATH=str(runtime.parent), PYTHONDONTWRITEBYTECODE="1",
               TRITON_CACHE_DIR=str(tmp_path / "cache"), HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    before = json.loads(subprocess.check_output([sys.executable, str(probe)], env=env, text=True, timeout=120))
    provider.write_text(original.replace(old, new))
    after = json.loads(subprocess.check_output([sys.executable, str(probe)], env=env, text=True, timeout=120))
    assert before[0] != after[0]
    assert before[1] != after[1]
    assert (before[2] == after[2]) == (source == "driver")
