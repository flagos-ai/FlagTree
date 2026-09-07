from triton.flagmega.compiler import Compiler
from triton.flagmega import ir as fm
from triton.flagmega.ir.printer import script_source


def _make_fp8_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    activation = fm.tensor_type("bfloat16", [1, 128])
    weight = fm.tensor_type("float8_e4m3fn", [128, 128])
    scale = fm.tensor_type("float32", [1, 1])
    source = builder.var("source", activation, id="source")
    rhs = builder.weight("rhs", weight, source="weights", key="rhs", id="rhs")
    rhs_scale = builder.weight("rhs_scale", scale, source="weights", key="rhs_scale", id="rhs_scale")
    result = builder.call(
        "math.block_scaled_matmul",
        [source, rhs, rhs_scale],
        activation,
        id="result",
        attrs={"weight_block_n": 128, "weight_block_k": 128},
    )
    builder.function("main", [source], [result])
    return builder.build(entry="main")


def test_bufferized_script_prints_physical_buffers_memspans_and_aliases():
    module = Compiler().compile(_make_fp8_module(), stop_after="bufferize").module

    source = script_source(module)

    assert "// physical buffers" in source
    assert "T.PhysicalBuffer(" in source
    assert "// logical buffer views" in source
    assert "MemSpan: T.MemSpan(" in source
    assert "T.Buffer(Dist(" in source
    assert "DistributedStorage: canonical_global" in source
    assert "PhysicalId" not in source
    assert "ByteOffset" not in source
