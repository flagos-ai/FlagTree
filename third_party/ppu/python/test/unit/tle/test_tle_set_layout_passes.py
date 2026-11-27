"""PPU explicit-layout pass regressions; no device or PPU SDK is required."""

import re

import pytest

from triton._C import libtriton

_LAYOUTS = """
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#c = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
"""


@pytest.fixture
def ppu_ir():
    if not hasattr(libtriton, "ppu"):
        pytest.skip("requires libtriton compiled with the PPU backend")
    assert hasattr(libtriton.ir.builder, "create_tle_gpu_set_layout"), "PPU build is missing the TLE layout builder"
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    libtriton.ppu.load_dialects(context)
    return context


def _parse(tmp_path, context, function):
    source = _LAYOUTS + """
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
""" + function + "\n}\n"
    path = tmp_path / "explicit-layout.ttgir"
    path.write_text(source)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    assert module.verify()
    return module


def _run(context, module, add_pass):
    manager = libtriton.ir.pass_manager(context)
    add_pass(manager)
    manager.run(module, "ppu_explicit_layout_regression")
    assert module.verify()


def _ops(module, name):
    operations = []
    module.walk(lambda op: operations.append(op) if op.get_name() == name else None)
    return operations


def _type(value):
    return re.sub(r"\s+", "", str(value.get_type()))


@pytest.mark.parametrize("explicit", [False, True])
def test_canonicalizer_preserves_explicit_conversion_boundary(tmp_path, ppu_ir, explicit):
    marker = '{"tle.explicit_encoding.0" = #b}' if explicit else ""
    module = _parse(
        tmp_path, ppu_ir, f"""
  tt.func @boundary(%x: tensor<256xf32, #a>) -> tensor<256xf32, #c> {{
    %hard = ttg.convert_layout %x {marker} : tensor<256xf32, #a> -> tensor<256xf32, #b>
    %out = ttg.convert_layout %hard : tensor<256xf32, #b> -> tensor<256xf32, #c>
    tt.return %out : tensor<256xf32, #c>
  }}
""")
    _run(ppu_ir, module, libtriton.passes.common.add_canonicalizer)
    converts = _ops(module, "ttg.convert_layout")
    assert len(converts) == (2 if explicit else 1), module.str_nodebug()
    if explicit:
        assert "sizePerThread=[2]" in _type(converts[0].get_result(0))
        assert "tle.explicit_encoding.0" in module.str_nodebug()


@pytest.mark.parametrize("explicit", [False, True])
def test_canonicalizer_preserves_explicit_splat_result(tmp_path, ppu_ir, explicit):
    marker = '{"tle.explicit_encoding.0" = #b}' if explicit else ""
    module = _parse(
        tmp_path, ppu_ir, f"""
  tt.func @splat(%x: f32) -> tensor<256xf32, #b> {{
    %splat = tt.splat %x : f32 -> tensor<256xf32, #a>
    %out = ttg.convert_layout %splat {marker} : tensor<256xf32, #a> -> tensor<256xf32, #b>
    tt.return %out : tensor<256xf32, #b>
  }}
""")
    _run(ppu_ir, module, libtriton.passes.common.add_canonicalizer)
    assert len(_ops(module, "ttg.convert_layout")) == int(explicit), module.str_nodebug()
    if explicit:
        assert "tle.explicit_encoding.0" in module.str_nodebug()


@pytest.mark.parametrize("conditional", [False, True])
def test_explicit_identity_boundary_survives_canonicalization(tmp_path, ppu_ir, conditional):
    producer = """
    %selected = scf.if %condition -> (tensor<256xf32, #b>) {
      scf.yield %x : tensor<256xf32, #b>
    } else {
      scf.yield %y : tensor<256xf32, #b>
    }
""" if conditional else ""
    value = "%selected" if conditional else "%x"
    module = _parse(
        tmp_path, ppu_ir, f"""
  tt.func @identity(%condition: i1, %x: tensor<256xf32, #b>, %y: tensor<256xf32, #b>) -> tensor<256xf32, #b> {{
    {producer}
    %out = ttg.convert_layout {value} {{"tle.explicit_encoding.0" = #b}} : tensor<256xf32, #b> -> tensor<256xf32, #b>
    tt.return %out : tensor<256xf32, #b>
  }}
""")
    _run(ppu_ir, module, libtriton.passes.common.add_canonicalizer)
    if conditional:
        assert not _ops(module, "scf.if")
        assert len(_ops(module, "arith.select")) == 1
    # PPU's pipeline runs layout cleanup more than once. The boundary must
    # survive producer canonicalization and each later layout-changing pass.
    for _ in range(2):
        _run(ppu_ir, module, libtriton.passes.ttgpuir.add_remove_layout_conversions)
        convert, = _ops(module, "ttg.convert_layout")
        assert _type(convert.get_result(0)) == _type(convert.get_operand(0))
        assert "sizePerThread=[2]" in _type(convert.get_result(0))
        assert "tle.explicit_encoding.0" in module.str_nodebug()


def _memory_function(operation, attributes):
    if operation == "tt.load":
        return f"""
  tt.func @load(%ptr: tensor<256x!tt.ptr<f32>, #b>) -> tensor<256xf32, #b> {{
    %value = tt.load %ptr {attributes} : tensor<256x!tt.ptr<f32>, #b>
    tt.return %value : tensor<256xf32, #b>
  }}
"""
    return f"""
  tt.func @store(%ptr: tensor<256x!tt.ptr<f32>, #b>, %value: tensor<256xf32, #b>) {{
    tt.store %ptr, %value {attributes} : tensor<256x!tt.ptr<f32>, #b>
    tt.return
  }}
"""


@pytest.mark.parametrize("operation", ["tt.load", "tt.store"])
@pytest.mark.parametrize("explicit", [False, True])
def test_coalesce_preserves_hard_memory_layout(tmp_path, ppu_ir, operation, explicit):
    marker_name = "tle.explicit_encoding.0" if operation == "tt.load" else "tle.explicit_memory_encoding"
    attributes = f'{{"{marker_name}" = #b}}' if explicit else ""
    module = _parse(tmp_path, ppu_ir, _memory_function(operation, attributes))
    original = _type(_ops(module, operation)[0].get_operand(0))
    _run(ppu_ir, module, libtriton.passes.ttgpuir.add_coalesce)
    memory_op, = _ops(module, operation)
    actual = _type(memory_op.get_operand(0))
    if explicit:
        assert actual == original, module.str_nodebug()
        assert marker_name in module.str_nodebug()
        data = memory_op.get_result(0) if operation == "tt.load" else memory_op.get_operand(1)
        assert "sizePerThread=[2]" in _type(data)
    else:
        # Tensor pointer arguments have no contiguity hints. The ordinary
        # coalescer picks one element per thread instead of the incoming two.
        assert actual != original, module.str_nodebug()
        assert "sizePerThread=[1]" in actual


@pytest.mark.parametrize("attributes,message", [
    ('{"tle.explicit_encoding.0" = #a}', "explicit TLE encoding that does not match the tensor type encoding"),
    ('{"tle.explicit_memory_encoding" = #a}',
     "explicit TLE memory encoding that does not match the pointer tensor encoding"),
    ('{"tle.explicit_encoding.0" = #b, "tle.explicit_memory_encoding" = #a}',
     "conflicting explicit TLE memory encodings"),
])
def test_coalesce_rejects_inconsistent_hard_layout(tmp_path, ppu_ir, capfd, attributes, message):
    module = _parse(tmp_path, ppu_ir, _memory_function("tt.load", attributes))
    manager = libtriton.ir.pass_manager(ppu_ir)
    libtriton.passes.ttgpuir.add_coalesce(manager)
    with pytest.raises(RuntimeError) as error:
        manager.run(module, "ppu_explicit_layout_rejection")
    diagnostics = capfd.readouterr().err
    assert message in str(error.value) + diagnostics
