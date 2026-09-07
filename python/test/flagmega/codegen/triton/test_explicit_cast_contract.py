from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.vectorization import vectorization_contract


def test_explicit_vector_cast_does_not_need_agent_selection_metadata():
    value = fm.Node("value", "builtin.var", (), fm.tensor_type(fm.vector_type("bfloat16", (8,)), (4,)),
                    attrs={"name": "value"})
    from triton.flagmega.rules.neutral._utility import make_node
    cast = make_node("ntt.vectorized_cast", "cast", (value,),
                     {"new_type": fm.vector_type("float32", (4,)), "vectorize_axes": (0,)}, {})
    contract = vectorization_contract(cast)
    assert contract["kind"] == "axes"
    assert contract["axes"] == (0,)
    assert contract["lanes"] == (4,)
