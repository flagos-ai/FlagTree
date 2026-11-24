from pathlib import Path

import triton.language as tl
import triton.tools.disasm as shared_disasm
import triton.tools.tensor_descriptor as tensor_descriptor
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.spec.xpu.triton.language import language_extend_globals


def test_shared_tool_module_is_not_loaded_from_xpu_spec():
    shared_path = Path(shared_disasm.__file__).resolve()
    assert shared_path.name == "disasm.py"
    assert shared_path.parent.name == "tools"
    assert "spec" not in shared_path.parts


def test_tensor_descriptor_uses_xpu_spec_and_accepts_unaligned_layout():
    assert Path(tensor_descriptor.__file__).parts[-6:] == (
        "triton",
        "spec",
        "xpu",
        "triton",
        "tools",
        "tensor_descriptor.py",
    )

    descriptor = TensorDescriptor(object(), [3, 513], [513, 1], [16, 32])
    assert descriptor.shape == [3, 513]


def test_language_exports_use_xpu_spec_init_hook():
    language_globals = {}
    language_extend_globals(language_globals)
    assert language_globals["atomic_mul"] is tl.atomic_mul
    assert language_globals["gelu"] is tl.gelu
