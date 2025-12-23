from triton.language.core import constexpr, builtin

from . import semantic


@builtin
def corex_sme(input, values, _builder=None):
    """
    Let the compiler know that the `input` tensor should use SME for the provided strides.

    e.g. if :code:`input` tensor has strides [64, 1], passing :code:`values = 64` enables SME load.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return semantic.corex_sme(input, values)
