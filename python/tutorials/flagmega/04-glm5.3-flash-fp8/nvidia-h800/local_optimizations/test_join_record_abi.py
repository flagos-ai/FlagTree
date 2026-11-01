"""Compile the actual join template's offset load with each typed record ABI."""
import ast
import importlib.util
from pathlib import Path

import pytest
from jinja2 import Environment, StrictUndefined
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@pytest.mark.parametrize("version,record_type", [(1, "*i64"), (2, "i64")])
def test_join_template_load_follows_typed_record_abi(version, record_type, tmp_path):
    template = Path(__file__).with_name("join-body.py.jinja").read_text()
    environment = Environment(undefined=StrictUndefined)
    definition = dict(owner_matched_join=dict(payload=0, ready=8192, consumed=8320),
                      function_pool_abi=dict(schema=f"flagmega.function-pool-abi/v{version}"))
    rendered = environment.from_string(template).render(definition=definition)
    tree = ast.parse("def body():\n" + rendered)
    setup = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "_flagmega_join_record_ptr"
                     for target in node.targets)]
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "load" and node.args
             and any(name in ast.unparse(node.args[0]) for name in
                     ("_flagmega_call_offsets", "_flagmega_join_record_ptr"))]
    assert len(loads) == 7
    statements = [ast.unparse(node) for node in setup]
    statements += ["value = " + ast.unparse(loads[0]), "tl.store(result, value)"]
    source = "import triton\nimport triton.language as tl\n@triton.jit\n"
    source += "def probe(_flagmega_function_offsets, _flagmega_call_offsets, result):\n"
    source += "\n".join("    " + line for line in statements) + "\n"
    path = tmp_path / "join_record_probe.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("join_record_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = dict(_flagmega_function_offsets="*i64", _flagmega_call_offsets=record_type, result="*i64")
    compiled = triton.compile(ASTSource(module.probe, signature), target=GPUTarget("cuda", 90, 32),
                              options=dict(num_warps=4, num_stages=1))
    assert compiled.metadata.ptxas_spill_load_bytes == compiled.metadata.ptxas_spill_store_bytes == 0
