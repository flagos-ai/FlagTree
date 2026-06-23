"""Generate Triton extern-call wrappers from CUDA device functions."""

import re
from dataclasses import dataclass
from pathlib import Path


_DTYPES = {
    "bool": "int1",
    "char": "int8",
    "signed char": "int8",
    "unsigned char": "uint8",
    "int8_t": "int8",
    "uint8_t": "uint8",
    "short": "int16",
    "short int": "int16",
    "signed short": "int16",
    "unsigned short": "uint16",
    "int16_t": "int16",
    "uint16_t": "uint16",
    "int": "int32",
    "signed": "int32",
    "signed int": "int32",
    "unsigned": "uint32",
    "unsigned int": "uint32",
    "int32_t": "int32",
    "uint32_t": "uint32",
    "long": "int64",
    "long int": "int64",
    "long long": "int64",
    "long long int": "int64",
    "unsigned long": "uint64",
    "unsigned long int": "uint64",
    "unsigned long long": "uint64",
    "unsigned long long int": "uint64",
    "int64_t": "int64",
    "uint64_t": "uint64",
    "size_t": "uint64",
    "float": "fp32",
    "double": "fp64",
}

_QUALIFIERS = re.compile(
    r"\b(?:const|volatile|restrict|__restrict__|__restrict|__const__|"
    r"__device__|__forceinline__|inline|static)\b"
)
_ATTRIBUTE = re.compile(
    r"__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)",
    re.DOTALL,
)
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_FUNCTION = re.compile(
    r'extern\s+"C"\s+(?P<prefix>.*?)\b(?P<name>[A-Za-z_]\w*)\s*'
    r"\((?P<params>.*?)\)\s*(?:noexcept\s*)?\{",
    re.DOTALL,
)


@dataclass(frozen=True)
class CudaType:
    dtype: str
    pointer: bool = False

    @property
    def triton_type(self) -> str:
        dtype = f'core.dtype("{self.dtype}")'
        return f"core.pointer_type({dtype})" if self.pointer else dtype


@dataclass(frozen=True)
class Parameter:
    name: str
    type: CudaType

    @property
    def argument(self) -> str:
        if self.type.pointer:
            return self.name
        return f"tl.cast({self.name}, tl.{self.type.dtype}, _semantic=_semantic)"


@dataclass(frozen=True)
class Function:
    name: str
    parameters: tuple[Parameter, ...]
    return_type: CudaType | None


def _split_parameters(parameters: str) -> list[str]:
    result = []
    start = 0
    depth = 0
    for index, char in enumerate(parameters):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(parameters[start:index].strip())
            start = index + 1
    tail = parameters[start:].strip()
    if tail and tail != "void":
        result.append(tail)
    return result


def _parse_type(spelling: str, *, context: str) -> CudaType | None:
    spelling = _ATTRIBUTE.sub(" ", spelling)
    pointer = "*" in spelling or "[" in spelling
    spelling = re.sub(r"\[[^\]]*\]", " ", spelling)
    spelling = spelling.replace("*", " ")
    spelling = _QUALIFIERS.sub(" ", spelling)
    spelling = " ".join(spelling.split())
    if spelling == "void" and not pointer:
        return None
    if spelling == "void" and pointer:
        return CudaType("uint8", pointer=True)
    dtype = _DTYPES.get(spelling)
    if dtype is None:
        raise ValueError(f"unsupported CUDA type {spelling!r} in {context}")
    return CudaType(dtype, pointer=pointer)


def _parse_parameter(parameter: str, function_name: str) -> Parameter:
    parameter = _ATTRIBUTE.sub(" ", parameter).strip()
    match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", parameter)
    if match is None:
        raise ValueError(f"cannot parse parameter {parameter!r} in {function_name}")
    name = match.group(1)
    type_spelling = parameter[:match.start(1)] + parameter[match.end(1):]
    parsed_type = _parse_type(type_spelling, context=f"{function_name}.{name}")
    if parsed_type is None:
        raise ValueError(f"parameter {name!r} in {function_name} cannot have type void")
    return Parameter(name, parsed_type)


def parse_cuda_functions(source: str) -> tuple[Function, ...]:
    source = _COMMENT.sub("", source)
    functions = []
    for match in _FUNCTION.finditer(source):
        if "__device__" not in match.group("prefix"):
            continue
        prefix = _QUALIFIERS.sub(" ", _ATTRIBUTE.sub(" ", match.group("prefix")))
        return_type = _parse_type(prefix, context=f"{match.group('name')} return type")
        parameters = tuple(
            _parse_parameter(parameter, match.group("name"))
            for parameter in _split_parameters(match.group("params"))
        )
        functions.append(Function(match.group("name"), parameters, return_type))
    return tuple(functions)


def _render_function(function: Function) -> str:
    names = ", ".join(parameter.name for parameter in function.parameters)
    signature = f"{names}, _semantic=None" if names else "_semantic=None"
    arguments = ",\n            ".join(parameter.argument for parameter in function.parameters)
    if arguments:
        arguments = f"[\n            {arguments},\n        ]"
    else:
        arguments = "[]"
    types = "\n".join(f"                {parameter.type.triton_type}," for parameter in function.parameters)
    returns = "()" if function.return_type is None else f"({function.return_type.triton_type},)"
    return (
        f"@core.extern\n"
        f"def {function.name}({signature}):\n"
        f"    return core.extern_call(\n"
        f'        "",\n'
        f'        "",\n'
        f"        {arguments},\n"
        f"        {{\n"
        f"            (\n{types}\n"
        f'            ): ("{function.name}", {returns}),\n'
        f"        }},\n"
        f"        is_pure=False,\n"
        f"        _semantic=_semantic,\n"
        f"    )\n"
    )


def generate(
    cuda_file: str | Path,
    output_file: str | Path,
    required_function: str | None = None,
) -> Path:
    cuda_file = Path(cuda_file)
    output_file = Path(output_file)
    functions = parse_cuda_functions(cuda_file.read_text(encoding="utf-8"))
    if not functions:
        raise ValueError(f"no extern \"C\" CUDA functions found in {cuda_file}")
    names = {function.name for function in functions}
    if required_function is not None and required_function not in names:
        raise ValueError(
            f"extern function {required_function!r} was not found in {cuda_file}; "
            f"found: {', '.join(sorted(names))}"
        )
    content = (
        "# Generated from "
        f"{cuda_file.name}; do not edit manually.\n"
        "import triton.language as tl\n"
        "import triton.language.core as core\n\n\n"
        + "\n\n".join(_render_function(function) for function in functions)
        + "\n"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not output_file.exists() or output_file.read_text(encoding="utf-8") != content:
        output_file.write_text(content, encoding="utf-8")
    return output_file
