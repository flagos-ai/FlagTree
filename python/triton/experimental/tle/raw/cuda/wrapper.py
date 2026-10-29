"""
Generate typed, fixed-size C++ wrappers before CUDA source optimization.

Floating-point values cross the wrapper boundary as same-width unsigned
bit patterns, and booleans as uint8_t. LLVM call lowering must implement
the matching bit casts and boolean conversions.
"""

import re


def _abi_(type_name):
    if type_name == "i1":
        return "bool", "uint8_t"
    widths = {
        "f16": 16,
        "bf16": 16,
        "f32": 32,
        "f64": 64,
        "f8E4M3FN": 8,
        "f8E4M3FNUZ": 8,
        "f8E5M2": 8,
        "f8E5M2FNUZ": 8,
    }
    if type_name in widths:
        return "float", f"uint{widths[type_name]}_t"
    if type_name and (type_name.startswith("f") or type_name.startswith("bf")):
        raise ValueError(f"unsupported floating-point raw wrapper type: {type_name}")
    return None, None


def generate_raw_wrappers(function_name, wrappers):
    if not re.fullmatch(r"(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*", function_name):
        raise ValueError("raw wrappers require a C++ function name, not a mangled symbol")
    parts = ["\n#include <stdint.h>\n#include <tuple>\n#include <type_traits>\n"]

    for wrapper in wrappers:
        symbol = wrapper["symbol"]
        namespace = f"{symbol}_types"
        # Derive non-fragment parameter types from the user's function, including pointer address spaces.
        declarations = [
            f"\nnamespace {namespace} {{", "template<class> struct signature;",
            "template<class R, class... A> struct signature<R(*)(A...)> {",
            "  using arguments = std::tuple<A...>; using result = R; };",
            "template<class R, class... A> struct signature<R(*)(A...) noexcept>", "  : signature<R(*)(A...)> {};",
            f"using fn = signature<decltype(&{function_name})>;",
            "template<int I> using arg = typename std::tuple_element<I, typename fn::arguments>::type;",
            "template<class To, class From>", "__device__ __forceinline__ To bit_cast(From value) {",
            '  static_assert(sizeof(To) == sizeof(From), "raw wrapper bit widths must match");',
            '  static_assert(std::is_trivially_copyable<To>::value && std::is_trivially_copyable<From>::value,',
            '                "raw wrapper bit casts require trivially copyable types");',
            "  return __builtin_bit_cast(To, value);", "}\n"
        ]

        parameters, setup, call_args, fields, returned = [], [], [], [], []
        native_index = 0
        for i, argument in enumerate(wrapper["arguments"]):
            kind = argument["kind"]
            if kind.startswith("tensor_fragment_"):
                count = argument["count"]
                if count <= 0:
                    raise ValueError("fragment must contain at least one element per thread")
                declarations.append(
                    f"using element_{i} = typename std::remove_cv<typename std::remove_pointer<arg<{native_index}>>::type>::type;"
                )
                conversion, wire_type = _abi_(argument.get("element_type"))
                native_type = f"{namespace}::element_{i}"
                if conversion == "bool":
                    declarations.append(
                        f'static_assert(std::is_same<element_{i}, bool>::value, "raw wrapper i1 requires bool");')
                parameter_type = wire_type or native_type
                for j in range(count):
                    parameters.append(f"{parameter_type} a{i}_{j}")
                values = []
                for j in range(count):
                    value = f"a{i}_{j}"
                    if conversion:
                        value = f"{namespace}::bit_cast<{native_type}>({value})"
                    values.append(value)
                setup.append(f"  {native_type} fragment_{i}[{count}] = {{{', '.join(values)}}};")
                call_args.extend((f"fragment_{i}", str(count)))
                if kind.endswith("_inout"):
                    for j in range(count):
                        fields.append(f"  {parameter_type} a{i}_{j};")
                        value = f"fragment_{i}[{j}]"
                        if conversion:
                            value = f"{namespace}::bit_cast<{wire_type}>({value})"
                        returned.append(value)
                native_index += 2
            else:
                width = 3 + 2 * argument["rank"] if "rank" in argument else 1
                # Only scalar values use value_type. A floating-point pointee or
                # memdesc element does not change the pointer/descriptor ABI.
                conversion, wire_type = _abi_(argument.get("value_type"))
                for j in range(width):
                    name = f"a{i}_{j}"
                    native_type = f"{namespace}::arg<{native_index}>"
                    parameters.append(f"{wire_type or native_type} {name}")
                    if conversion:
                        call_args.append(f"{namespace}::bit_cast<{native_type}>({name})")
                        if conversion == "bool":
                            declarations.append(
                                f'static_assert(std::is_same<typename std::remove_cv<arg<{native_index}>>::type, bool>::value, '
                                '"raw wrapper i1 requires bool");')
                    else:
                        call_args.append(name)
                    native_index += 1
        declarations.append("}\n")

        parts.extend(declarations)
        return_type = "void"
        if fields:
            return_type = f"{symbol}_result"
            parts.extend([f"struct {return_type} {{", *fields, "};"])
        parts.append(f'extern "C" __device__ {return_type} {symbol}({", ".join(parameters)}) {{')
        parts.extend(setup)
        # Force the original device function into this wrapper while Clang sees
        # the literal fragment lengths, rather than deferring this to Triton.
        parts.append(f"  [[clang::always_inline]] {function_name}({', '.join(call_args)});")
        if fields:
            parts.append(f"  return {{{', '.join(returned)}}};")
        parts.append("}\n")
    return "\n".join(parts) + "\n"
