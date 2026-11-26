def normalize_musa_arch(arch: object) -> str:
    arch_str = str(arch).lower()
    if "." in arch_str:
        arch_major, arch_minor = arch_str.split(".", 1)
        if arch_major.isdecimal() and arch_minor.isdecimal():
            arch_str = f"{arch_major}{arch_minor}"

    # Canonicalize product and processor aliases before they cross the Python
    # backend boundary.  The C++ passes intentionally consume only numeric
    # capabilities, so leaving aliases here can silently bypass SQMMA
    # selection and fall back to scalar FMA lowering.
    if arch_str in {"mp32", "mp_32"} or arch_str.startswith("ph1s"):
        return "32"
    if arch_str in {"mp31", "mp_31"} or arch_str.startswith("ph1"):
        return "31"
    return arch_str


def musa_capability_from_arch(arch: object) -> int | None:
    arch_str = normalize_musa_arch(arch)
    if arch_str.isdecimal():
        return int(arch_str)
    return None


def musa_warp_size_from_arch(arch: object) -> int:
    capability = musa_capability_from_arch(arch)
    return 128 if capability is not None and capability < 30 else 32


musa_capability_from_arch.__triton_builtin__ = True
