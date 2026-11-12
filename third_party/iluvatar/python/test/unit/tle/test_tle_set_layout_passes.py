"""Iluvatar explicit-layout pass regressions; no device or Iluvatar SDK is required."""

import re
from pathlib import Path

import pytest

from triton._C import libtriton
from triton._flagtree_backend import FLAGTREE_BACKEND

_LAYOUTS = """
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#c = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
"""


@pytest.fixture
def iluvatar_ir():
    if FLAGTREE_BACKEND != "iluvatar":
        pytest.skip("requires libtriton compiled with the Iluvatar backend")
    assert hasattr(libtriton, "iluvatar")
    assert hasattr(libtriton.ir.builder,
                   "create_tle_gpu_set_layout"), "Iluvatar build is missing the TLE layout builder"
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    libtriton.iluvatar.load_dialects(context)
    return context


def _native_source():
    return """
#layout = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module {
  tt.func @layout(%ptr: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %pointers = tt.addptr %base, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %x = tt.load %pointers : tensor<256x!tt.ptr<f32>>
    %out = "iluvatar_tle.gpu.set_layout"(%x) {target_encoding = #layout} : (tensor<256xf32>) -> tensor<256xf32>
    tt.store %pointers, %out : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
"""


def test_native_set_layout_conversion(tmp_path, iluvatar_ir):
    path = tmp_path / "set-layout.ttir"
    path.write_text(_native_source())
    module = libtriton.ir.parse_mlir_module(str(path), iluvatar_ir)
    assert module.verify()
    _run(iluvatar_ir, module, lambda pm: libtriton.passes.ttir.add_convert_to_ttgpuir(pm, "cuda:71", 4, 64, 1))
    assert '"ttg.threads-per-warp" = 64' in module.str_nodebug()
    assert not _ops(module, "iluvatar_tle.gpu.set_layout")
    store, = _ops(module, "tt.store")
    assert "threadsPerWarp=[64]" in _type(store.get_operand(1))
    assert "sizePerThread=[2]" in _type(store.get_operand(1))


def test_native_set_layout_rejects_multi_cta_context(tmp_path, iluvatar_ir, capfd):
    # CorexOptions rejects num_ctas > 1 before the pass. Check the pass itself
    # as well, so users compiling TTIR cannot bypass the set_layout limit.
    path = tmp_path / "multi-cta.ttir"
    path.write_text(_native_source())
    module = libtriton.ir.parse_mlir_module(str(path), iluvatar_ir)
    manager = libtriton.ir.pass_manager(iluvatar_ir)
    libtriton.passes.ttir.add_convert_to_ttgpuir(manager, "cuda:71", 4, 64, 2)
    with pytest.raises(RuntimeError) as error:
        manager.run(module, "iluvatar_explicit_layout_multi_cta")
    assert "requires num_ctas = 1" in str(error.value) + capfd.readouterr().err


def test_native_set_layout_rejects_tensor_function_boundary(tmp_path, iluvatar_ir, capfd):
    path = tmp_path / "tensor-abi.ttir"
    path.write_text("""
#layout = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module {
  tt.func private @helper(%x: tensor<256xf32>) -> tensor<256xf32> {
    %out = "iluvatar_tle.gpu.set_layout"(%x) {target_encoding = #layout} : (tensor<256xf32>) -> tensor<256xf32>
    tt.return %out : tensor<256xf32>
  }
}
""")
    module = libtriton.ir.parse_mlir_module(str(path), iluvatar_ir)
    manager = libtriton.ir.pass_manager(iluvatar_ir)
    libtriton.passes.ttir.add_convert_to_ttgpuir(manager, "cuda:71", 4, 64, 1)
    with pytest.raises(RuntimeError) as error:
        manager.run(module, "iluvatar_explicit_layout_tensor_abi")
    assert "requires tensor function boundaries to be inlined" in str(error.value) + capfd.readouterr().err


def test_native_set_layout_rejects_linear_layout(tmp_path, iluvatar_ir, capfd):
    source = _native_source()
    linear = "#ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [[64], [128]], block = []}>"
    source = re.sub(r"#layout = .*", f"#layout = {linear}", source)
    path = tmp_path / "unsupported-linear.ttir"
    path.write_text(source)
    # In particular, rejection must not invoke unsupported getRepOrder() on
    # LinearEncodingAttr, which aborts the compiler process instead of raising.
    with pytest.raises(RuntimeError) as error:
        libtriton.ir.parse_mlir_module(str(path), iluvatar_ir)
    assert "only supports BlockEncoding and SlicedEncoding" in str(error.value) + capfd.readouterr().err


@pytest.mark.parametrize("cga_layout", [[[1]], [[-1]], [[1, 2]]])
def test_blocked_builder_rejects_nonempty_cga_layout(iluvatar_ir, cga_layout):
    builder = libtriton.ir.builder(iluvatar_ir)
    with pytest.raises(ValueError, match="requires num_ctas = 1 and an empty cga_layout"):
        builder.get_blocked_encoding([1], [64], [4], [0], cga_layout)


def _parse(tmp_path, context, function):
    source = _LAYOUTS + """
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:71", "ttg.threads-per-warp" = 64 : i32} {
""" + function + "\n}\n"
    path = tmp_path / "explicit-layout.ttgir"
    path.write_text(source)
    module = libtriton.ir.parse_mlir_module(str(path), context)
    assert module.verify()
    return module


def _run(context, module, add_pass):
    manager = libtriton.ir.pass_manager(context)
    add_pass(manager)
    manager.run(module, "iluvatar_explicit_layout_regression")
    assert module.verify()


def _ops(module, name):
    operations = []
    module.walk(lambda op: operations.append(op) if op.get_name() == name else None)
    return operations


def _type(value):
    return re.sub(r"\s+", "", str(value.get_type()))


_FIXTURE_CASES = [(name, index)
                  for name in ("layout_anchors", "thread_locality")
                  for index, _ in enumerate((Path(__file__).parent / "inputs" /
                                             f"set_layout_{name}.mlir").read_text().split("// -----"))]


@pytest.mark.parametrize("fixture,index", _FIXTURE_CASES)
def test_hard_layout_pass_fixtures(tmp_path, iluvatar_ir, fixture, index):
    source = (Path(__file__).parent / "inputs" / f"set_layout_{fixture}.mlir").read_text()
    path = tmp_path / f"{fixture}-{index}.ttgir"
    path.write_text(source.split("// -----")[index])
    module = libtriton.ir.parse_mlir_module(str(path), iluvatar_ir)
    assert module.verify()
    expected_ops = {
        "layout_anchors": [("arith.addf", ), ("arith.addf", ), (), ("arith.addf", "ttg.convert_layout")],
        "thread_locality": [("tt.reshape", ), ("tt.gather", ), ("tt.reduce", ), ("arith.addf", )],
    }[fixture][index]

    def hard_results():
        # Compare the actual result type before/after the optimization. Merely
        # retaining the attribute while rewriting its tensor type is incorrect.
        result = []
        for name in expected_ops:
            for op in _ops(module, name):
                result_type = _type(op.get_result(0))
                if result_type.startswith("tensor<"):
                    result.append((name, result_type))
        return sorted(result)

    original = hard_results()
    marker_count = module.str_nodebug().count("tle.explicit_encoding.0")
    if fixture == "layout_anchors":
        add_pass = libtriton.passes.ttgpuir.add_remove_layout_conversions
    else:
        add_pass = libtriton.passes.ttgpuir.add_optimize_thread_locality
    _run(iluvatar_ir, module, add_pass)
    if fixture == "layout_anchors" and index == 2:
        assert not original
        assert not _ops(module, "ttg.convert_layout"), module.str_nodebug()
    else:
        assert original, "fixture must contain a hard layout constraint"
        assert marker_count == len(original)
        assert module.str_nodebug().count("tle.explicit_encoding.0") == marker_count
        assert hard_results() == original, module.str_nodebug()


@pytest.mark.parametrize("explicit", [False, True])
def test_canonicalizer_preserves_explicit_conversion_boundary(tmp_path, iluvatar_ir, explicit):
    marker = '{"tle.explicit_encoding.0" = #b}' if explicit else ""
    module = _parse(
        tmp_path, iluvatar_ir, f"""
  tt.func @boundary(%x: tensor<256xf32, #a>) -> tensor<256xf32, #c> {{
    %hard = ttg.convert_layout %x {marker} : tensor<256xf32, #a> -> tensor<256xf32, #b>
    %out = ttg.convert_layout %hard : tensor<256xf32, #b> -> tensor<256xf32, #c>
    tt.return %out : tensor<256xf32, #c>
  }}
""")
    _run(iluvatar_ir, module, libtriton.passes.common.add_canonicalizer)
    converts = _ops(module, "ttg.convert_layout")
    assert len(converts) == (2 if explicit else 1), module.str_nodebug()
    if explicit:
        assert "sizePerThread=[2]" in _type(converts[0].get_result(0))
        assert "tle.explicit_encoding.0" in module.str_nodebug()


@pytest.mark.parametrize("explicit", [False, True])
def test_canonicalizer_preserves_explicit_splat_result(tmp_path, iluvatar_ir, explicit):
    marker = '{"tle.explicit_encoding.0" = #b}' if explicit else ""
    module = _parse(
        tmp_path, iluvatar_ir, f"""
  tt.func @splat(%x: f32) -> tensor<256xf32, #b> {{
    %splat = tt.splat %x : f32 -> tensor<256xf32, #a>
    %out = ttg.convert_layout %splat {marker} : tensor<256xf32, #a> -> tensor<256xf32, #b>
    tt.return %out : tensor<256xf32, #b>
  }}
""")
    _run(iluvatar_ir, module, libtriton.passes.common.add_canonicalizer)
    assert len(_ops(module, "ttg.convert_layout")) == int(explicit), module.str_nodebug()
    if explicit:
        assert "tle.explicit_encoding.0" in module.str_nodebug()


@pytest.mark.parametrize("explicit", [False, True])
def test_canonicalizer_preserves_explicit_range_producer(tmp_path, iluvatar_ir, explicit):
    marker = ', "tle.explicit_encoding.0" = #b' if explicit else ""
    module = _parse(
        tmp_path, iluvatar_ir, f"""
  tt.func @range() -> tensor<256xi32, #a> {{
    %range = tt.make_range {{start = 0 : i32, end = 256 : i32{marker}}} : tensor<256xi32, #b>
    %out = ttg.convert_layout %range : tensor<256xi32, #b> -> tensor<256xi32, #a>
    tt.return %out : tensor<256xi32, #a>
  }}
""")
    _run(iluvatar_ir, module, libtriton.passes.common.add_canonicalizer)
    producer, = _ops(module, "tt.make_range")
    assert len(_ops(module, "ttg.convert_layout")) == int(explicit), module.str_nodebug()
    expected_size = 2 if explicit else 1
    assert f"sizePerThread=[{expected_size}]" in _type(producer.get_result(0)), module.str_nodebug()
    if explicit:
        assert "tle.explicit_encoding.0" in module.str_nodebug()


@pytest.mark.parametrize("conditional", [False, True])
def test_explicit_identity_boundary_survives_canonicalization(tmp_path, iluvatar_ir, conditional):
    producer = """
    %selected = scf.if %condition -> (tensor<256xf32, #b>) {
      scf.yield %x : tensor<256xf32, #b>
    } else {
      scf.yield %y : tensor<256xf32, #b>
    }
""" if conditional else ""
    value = "%selected" if conditional else "%x"
    module = _parse(
        tmp_path, iluvatar_ir, f"""
  tt.func @identity(%condition: i1, %x: tensor<256xf32, #b>, %y: tensor<256xf32, #b>) -> tensor<256xf32, #b> {{
    {producer}
    %out = ttg.convert_layout {value} {{"tle.explicit_encoding.0" = #b}} : tensor<256xf32, #b> -> tensor<256xf32, #b>
    tt.return %out : tensor<256xf32, #b>
  }}
""")
    _run(iluvatar_ir, module, libtriton.passes.common.add_canonicalizer)
    if conditional:
        assert not _ops(module, "scf.if")
        assert len(_ops(module, "arith.select")) == 1
    # Iluvatar's pipeline runs layout cleanup more than once. The boundary must
    # survive producer canonicalization and each later layout-changing pass.
    for _ in range(2):
        _run(iluvatar_ir, module, libtriton.passes.ttgpuir.add_remove_layout_conversions)
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
def test_coalesce_preserves_hard_memory_layout(tmp_path, iluvatar_ir, operation, explicit):
    marker_name = "tle.explicit_encoding.0" if operation == "tt.load" else "tle.explicit_memory_encoding"
    attributes = f'{{"{marker_name}" = #b}}' if explicit else ""
    module = _parse(tmp_path, iluvatar_ir, _memory_function(operation, attributes))
    original = _type(_ops(module, operation)[0].get_operand(0))
    _run(iluvatar_ir, module, libtriton.passes.ttgpuir.add_coalesce)
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


@pytest.mark.parametrize("explicit", [False, True])
def test_local_pointer_rewrites_preserve_same_layout(tmp_path, iluvatar_ir, explicit):
    shared_type = ("!ttg.memdesc<256xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, "
                   "#ttg.shared_memory, mutable>")
    store_marker = '{"tle.explicit_memory_encoding" = #b}' if explicit else ""
    load_marker = '{"tle.explicit_encoding.0" = #b}' if explicit else ""
    module = _parse(
        tmp_path, iluvatar_ir, f"""
  tt.func @local(%out: tensor<256x!tt.ptr<f32>, #b>, %value: tensor<256xf32, #b>) {{
    %buf = ttg.local_alloc : () -> {shared_type}
    %idx = tt.make_range {{end = 256 : i32, start = 0 : i32}} : tensor<256xi32, #b>
    %ptrs = "iluvatar_tle.local_pointers"(%buf, %idx) : ({shared_type}, tensor<256xi32, #b>) -> tensor<256x!tt.ptr<f32, 3>, #b>
    tt.store %ptrs, %value {store_marker} : tensor<256x!tt.ptr<f32, 3>, #b>
    %x = tt.load %ptrs {load_marker} : tensor<256x!tt.ptr<f32, 3>, #b>
    tt.store %out, %x : tensor<256x!tt.ptr<f32>, #b>
    tt.return
  }}
""")
    load_type = _type(_ops(module, "tt.load")[0].get_result(0))
    store_type = _type(_ops(module, "tt.store")[0].get_operand(1))

    def add_local_pointer_passes(manager):
        libtriton.iluvatar.tle.passes.add_optimize_local_pointer_loads(manager)
        libtriton.iluvatar.tle.passes.add_optimize_local_pointer_stores(manager)

    _run(iluvatar_ir, module, add_local_pointer_passes)
    # These rewrites retain the distributed layout. An explicit constraint
    # must survive the rewrite without disabling the ordinary optimization.
    for add_pass in (libtriton.passes.common.add_canonicalizer, libtriton.passes.ttgpuir.add_remove_layout_conversions):
        _run(iluvatar_ir, module, add_pass)
        load, = _ops(module, "ttg.local_load")
        store, = _ops(module, "ttg.local_store")
        assert _type(load.get_result(0)) == load_type, module.str_nodebug()
        assert _type(store.get_operand(0)) == store_type, module.str_nodebug()
        assert not _ops(module, "tt.load"), module.str_nodebug()
        if explicit:
            lines = module.str_nodebug().splitlines()
            load_line, = [line for line in lines if "ttg.local_load " in line]
            store_line, = [line for line in lines if "ttg.local_store " in line]
            assert "tle.explicit_encoding.0" in load_line, load_line
            assert "tle.explicit_memory_encoding" in load_line, load_line
            assert "tle.explicit_memory_encoding" in store_line, store_line


@pytest.mark.parametrize("attributes,message", [
    ('{"tle.explicit_encoding.0" = #a}', "explicit TLE encoding that does not match the tensor type encoding"),
    ('{"tle.explicit_memory_encoding" = #a}',
     "explicit TLE memory encoding that does not match the pointer tensor encoding"),
    ('{"tle.explicit_encoding.0" = #b, "tle.explicit_memory_encoding" = #a}',
     "conflicting explicit TLE memory encodings"),
])
def test_coalesce_rejects_inconsistent_hard_layout(tmp_path, iluvatar_ir, capfd, attributes, message):
    module = _parse(tmp_path, iluvatar_ir, _memory_function("tt.load", attributes))
    manager = libtriton.ir.pass_manager(iluvatar_ir)
    libtriton.passes.ttgpuir.add_coalesce(manager)
    with pytest.raises(RuntimeError) as error:
        manager.run(module, "iluvatar_explicit_layout_rejection")
    diagnostics = capfd.readouterr().err
    assert message in str(error.value) + diagnostics


@pytest.mark.parametrize("pass_name,operation,options", [
    ("add_optimize_local_pointer_loads", "tt.load", ()),
    ("add_optimize_local_pointer_stores", "tt.store", ()),
    ("add_optimize_local_pointer_async_stores", "tt.store", (0, 1)),
])
def test_local_pointer_passes_reject_inconsistent_hard_layout(tmp_path, iluvatar_ir, capfd, pass_name, operation,
                                                              options):
    if operation == "tt.load":
        function = _memory_function(operation, '{"tle.explicit_encoding.0" = #a}')
    else:
        function = """
  tt.func @store(%ptr: tensor<256x!tt.ptr<f32>, #b>, %scalar: f32) {
    %value = tt.splat %scalar {"tle.explicit_encoding.0" = #a} : f32 -> tensor<256xf32, #b>
    tt.store %ptr, %value : tensor<256x!tt.ptr<f32>, #b>
    tt.return
  }
"""
    module = _parse(tmp_path, iluvatar_ir, function)
    manager = libtriton.ir.pass_manager(iluvatar_ir)
    getattr(libtriton.iluvatar.tle.passes, pass_name)(manager, *options)
    # Printing an error and returning success would let compilation continue
    # with an invalid hard layout. The pass manager must report failure.
    with pytest.raises(RuntimeError) as error:
        manager.run(module, "iluvatar_explicit_layout_rejection")
    assert "explicit TLE encoding that does not match the tensor type encoding" in (str(error.value) +
                                                                                    capfd.readouterr().err)
