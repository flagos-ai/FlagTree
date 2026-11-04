# flagtree tle
"""Structural pipe types must not specialize functions by diagnostic names."""

from triton.experimental.tle.language.gpu.types import pipe_value_type


def test_pipe_names_do_not_change_function_specialization():
    first = pipe_value_type(2, "cta", "layer_0", [], one_shot=True)
    second = pipe_value_type(2, "cta", "layer_1", [], one_shot=True)
    assert first == second
    assert first.mangle() == second.mangle()


def test_fieldless_pipe_has_an_explicit_dynamic_identity():
    identity = object()
    ty = pipe_value_type(1, "cta", None, [], one_shot=True)
    value, cursor = ty._unflatten_ir([identity], 0)
    flattened = []
    value._flatten_ir(flattened)
    assert cursor == 1
    assert flattened == [identity]


def test_protocol_is_part_of_the_structural_type():
    cyclic = pipe_value_type(1, "cta", None, [])
    one_shot = pipe_value_type(1, "cta", None, [], one_shot=True)
    assert cyclic != one_shot
    assert cyclic.mangle() != one_shot.mangle()
