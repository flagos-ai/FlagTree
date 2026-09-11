def normalize_musa_arch(arch: object) -> str:
    arch_str = str(arch).lower()
    if "." not in arch_str:
        return arch_str

    arch_major, arch_minor = arch_str.split(".", 1)
    if arch_major.isdecimal() and arch_minor.isdecimal():
        return f"{arch_major}{arch_minor}"
    return arch_str


def musa_capability_from_arch(arch: object) -> int | None:
    arch_str = normalize_musa_arch(arch)
    if arch_str.startswith("ph1"):
        return 31
    if arch_str.isdecimal():
        return int(arch_str)
    return None


def musa_warp_size_from_arch(arch: object) -> int:
    capability = musa_capability_from_arch(arch)
    return 128 if capability is not None and capability < 30 else 32


musa_capability_from_arch.__triton_builtin__ = True
