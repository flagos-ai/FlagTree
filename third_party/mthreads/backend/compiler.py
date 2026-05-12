from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, mthreads
from triton import knobs

from dataclasses import dataclass
from pathlib import Path
import functools
from typing import Any, Dict, Tuple, Optional
import hashlib
import os
import re
import shutil
import shlex
import subprocess
import tempfile


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        return (1, 1, 1)

    return check_dot_compatibility


def _module_text(mod) -> str:
    try:
        return str(mod)
    except Exception:
        return ""


def _module_uses_sqmma(mod) -> bool:
    text = _module_text(mod)
    return "mtgpu.sqmma" in text


@functools.lru_cache()
def get_musa_version() -> str:
    if env_ver := os.getenv("TRITON_MUSA_VERSION"):
        return env_ver
    try:
        import torch_musa  # type: ignore
        return getattr(torch_musa, "__version__", "unknown")
    except Exception:
        return "unknown"


@functools.lru_cache(None)
def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@functools.lru_cache(None)
def _tool_version_signature(path: str) -> str:
    norm = _normalize_path(path)
    if not norm:
        return ""
    tool_path = str(Path(norm).expanduser())
    version_text = ""
    try:
        out = subprocess.check_output([tool_path, "--version"], stderr=subprocess.STDOUT, text=True)
        version_text = out.strip()
    except Exception:
        version_text = ""
    binary_hash = ""
    try:
        if Path(tool_path).exists():
            binary_hash = file_hash(tool_path)
    except Exception:
        binary_hash = ""
    return f"{tool_path}|{version_text}|{binary_hash}"


def _normalize_arch(arch: object) -> str:
    if isinstance(arch, int):
        return str(arch)
    return str(arch).lower()


def _capability_from_arch(arch: object) -> int:
    if isinstance(arch, int):
        return arch
    arch_str = _normalize_arch(arch)
    if arch_str.isdigit():
        return int(arch_str)
    if arch_str.startswith("ph1"):
        return 31
    raise ValueError(f"Unsupported MUSA arch: {arch}")


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return str(Path(path).expanduser())


def _maybe_tool_path(tool) -> Optional[str]:
    try:
        return _normalize_path(tool.path)
    except Exception:
        return None


def _select_tool_path(explicit_path: Optional[str], tool) -> Optional[str]:
    path = _normalize_path(explicit_path)
    if path:
        return path
    return _maybe_tool_path(tool)


def _resolve_toolchain_paths(options: "MUSAOptions") -> Tuple[str, str, Optional[str]]:
    toolchain_path = _normalize_path(options.toolchain_path)
    llc_path = _normalize_path(options.llc_path)
    lld_path = _normalize_path(options.lld_path)
    llc_asm_path = _normalize_path(options.llc_asm_path)

    if not toolchain_path:
        mtcc_bin_path = os.getenv("MTCC_BIN_PATH")
        if mtcc_bin_path:
            toolchain_path = str(Path(mtcc_bin_path).expanduser())
    if not toolchain_path:
        musa_home = os.getenv("MUSA_HOME")
        if musa_home:
            toolchain_path = str(Path(musa_home).expanduser() / "bin")

    if not llc_path and toolchain_path:
        llc_path = str(Path(toolchain_path) / "llc")
    if not lld_path and toolchain_path:
        lld_path = str(Path(toolchain_path) / "ld.lld")

    return llc_path or "", lld_path or "", llc_asm_path


@functools.lru_cache(None)
def _detect_llvm_major_version(llc_path: str) -> Optional[int]:
    llc = str(Path(llc_path).expanduser()) if llc_path else ""
    if not llc:
        return None
    try:
        out = subprocess.check_output([llc, "--version"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None
    match = re.search(r"LLVM version\s+(\d+)\.", out)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _tool_output(stdout: Optional[str], stderr: Optional[str]) -> str:
    chunks = []
    if stdout and stdout.strip():
        chunks.append(stdout.strip())
    if stderr and stderr.strip():
        chunks.append(stderr.strip())
    return "\n".join(chunks)


def _run_tool_command(tool_name: str, cmd: list[str], *, repro_dir: Path, dump_log: bool = False) -> None:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    output = _tool_output(proc.stdout, proc.stderr)
    if dump_log and output:
        print(f"// -----// MUSA {tool_name} Log //----- //")
        print(output)
    if proc.returncode == 0:
        return
    error = (f"`{tool_name}` failed with error code {proc.returncode}\n"
             f"`{tool_name}` output:\n{output or '<empty>'}\n"
             f"Repro command: {shlex.join(cmd)}\n"
             f"Artifacts kept in: {repro_dir}")
    raise RuntimeError(error)


def _should_apply_llvm_compat(llc_major: Optional[int]) -> bool:
    return llc_major is None or llc_major < 19


def _llc_opaque_pointer_options(llc_major: Optional[int]) -> list[str]:
    return ["--opaque-pointers"] if llc_major is not None and llc_major < 15 else []


def _strip_range_attributes(ir_text: str) -> str:
    out = ir_text
    pos = 0
    call_ret_re = re.compile(r"[^,\n@][^,\n@]*\s+@[A-Za-z_$.][A-Za-z0-9_$.]*\s*\(")
    while True:
        start = out.find("range(", pos)
        if start < 0:
            break
        cur = start + len("range(")
        depth = 1
        while cur < len(out) and depth > 0:
            ch = out[cur]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            cur += 1
        if depth != 0:
            pos = start + 1
            continue
        end = cur
        while end < len(out) and out[end].isspace():
            end += 1
        tail = out[end:]
        if end < len(out) and (out[end] == "%" or call_ret_re.match(tail)):
            out = out[:start] + out[end:]
            pos = start
        else:
            pos = end
    return out


def _rewrite_bare_splat_operands(ir_text: str) -> str:
    vec_re = re.compile(r"<\s*(\d+)\s+x\s*([A-Za-z0-9_.]+)\s*>")
    bare_splat_re = re.compile(r"splat\s*\(\s*([A-Za-z0-9_.]+)\s+([^)]+)\s*\)")

    out_lines = []
    for line in ir_text.splitlines():
        search_pos = 0
        while search_pos < len(line):
            match = bare_splat_re.search(line, search_pos)
            if match is None:
                break
            elem_ty = match.group(1)
            elem_val = match.group(2).strip()
            prefix = line[:match.start()]

            lane_count = -1
            for vec_match in vec_re.finditer(prefix):
                if vec_match.group(2) == elem_ty:
                    lane_count = int(vec_match.group(1))

            if lane_count <= 0:
                search_pos = match.end()
                continue

            lane_str = str(lane_count)
            vec_ty = f"<{lane_str} x {elem_ty}>"
            mask_ty = f"<{lane_str} x i32>"
            insert_expr = (f"insertelement ({vec_ty} undef, {elem_ty} {elem_val}, i32 0)")
            replacement = (f"shufflevector ({vec_ty} {insert_expr}, {vec_ty} undef, "
                           f"{mask_ty} zeroinitializer)")

            line = line[:match.start()] + replacement + line[match.end():]
            search_pos = match.start() + len(replacement)

        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if ir_text.endswith("\n") else "")


def _rewrite_musa_isspacep_shared(ir_text: str) -> str:
    call_re = re.compile(
        r"^([ \t]*)(%[A-Za-z0-9_.]+|%\d+)\s*=\s*(?:tail\s+)?call\s+i1\s+"
        r"@llvm\.musa\.isspacep\.shared\s*\(\s*ptr(?:\s+[^()%]+)*\s+(%[A-Za-z0-9_.]+|%\d+)\s*\)\s*(,.*)?$")

    def _tmp_name(base_pred: str, kind: str) -> str:
        if re.fullmatch(r"%\d+", base_pred):
            return f"%musa_isspacep_{kind}_{base_pred[1:]}"
        return f"{base_pred}.isspacep.{kind}"

    out_lines = []
    for line in ir_text.splitlines():
        m = call_re.match(line)
        if m is None:
            out_lines.append(line)
            continue

        indent, pred_name, ptr_name, dbg_suffix = m.groups()
        dbg_suffix = dbg_suffix or ""
        ptr_i64 = _tmp_name(pred_name, "i64")
        ptr_hi32 = _tmp_name(pred_name, "hi32")
        out_lines.append(f"{indent}{ptr_i64} = ptrtoint ptr {ptr_name} to i64{dbg_suffix}")
        out_lines.append(f"{indent}{ptr_hi32} = lshr i64 {ptr_i64}, 32{dbg_suffix}")
        out_lines.append(f"{indent}{pred_name} = icmp eq i64 {ptr_hi32}, 0{dbg_suffix}")

    out = "\n".join(out_lines)
    if ir_text.endswith("\n"):
        out += "\n"

    out = re.sub(
        r"(?m)^[ \t]*declare\s+i1\s+@llvm\.musa\.isspacep\.shared\s*\(\s*ptr\s*\)\s*(?:#\d+)?\s*\n?",
        "",
        out,
    )
    return out


def _rewrite_musa_ptr_gen_to_addrspace(ir_text: str) -> str:
    specs = [("global", 1), ("shared", 3)]
    ptr_as_map: Dict[str, int] = {}
    out_lines = []

    for line in ir_text.splitlines():
        rewritten = False
        for space_name, as_id in specs:
            call_re = re.compile(
                rf"^([ \t]*)(%[A-Za-z0-9_.]+|%\d+)\s*=\s*(?:tail\s+)?call\s+ptr\s+"
                rf"@llvm\.musa\.ptr\.gen\.to\.{space_name}\s*\(\s*ptr(?:\s+[^()%]+)*\s+(%[A-Za-z0-9_.]+|%\d+)\s*\)\s*(,.*)?$"
            )
            m = call_re.match(line)
            if m is None:
                continue
            indent, out_ptr, in_ptr, dbg_suffix = m.groups()
            dbg_suffix = dbg_suffix or ""
            out_lines.append(f"{indent}{out_ptr} = addrspacecast ptr {in_ptr} to ptr addrspace({as_id}){dbg_suffix}")
            ptr_as_map[out_ptr] = as_id
            rewritten = True
            break
        if not rewritten:
            out_lines.append(line)

    out = "\n".join(out_lines)
    if ir_text.endswith("\n"):
        out += "\n"
    for space_name, _ in specs:
        out = re.sub(
            rf"(?m)^[ \t]*declare\s+ptr\s+@llvm\.musa\.ptr\.gen\.to\.{space_name}\s*\(\s*ptr\s*\)\s*(?:#\d+)?\s*\n?",
            "",
            out,
        )

    for ptr_name, as_id in ptr_as_map.items():
        out = re.sub(
            rf"\bcmpxchg\s+ptr\s+{re.escape(ptr_name)}\b",
            f"cmpxchg ptr addrspace({as_id}) {ptr_name}",
            out,
        )
    return out


def _rewrite_llvm_is_fpclass_f32(ir_text: str) -> str:
    call_re = re.compile(r"^([ \t]*)(%[A-Za-z0-9_.]+|%\d+)\s*=\s*(?:tail\s+)?call\s+i1\s+"
                         r"@llvm\.is\.fpclass\.f32\s*\(\s*float\s+([^,]+)\s*,\s*i32\s+64\s*\)\s*(,.*)?$")
    out_lines = []
    changed = False
    for line in ir_text.splitlines():
        m = call_re.match(line)
        if m is None:
            out_lines.append(line)
            continue
        indent, pred_name, val, dbg_suffix = m.groups()
        dbg_suffix = dbg_suffix or ""
        out_lines.append(f"{indent}{pred_name} = fcmp oeq float {val.strip()}, 0.000000e+00{dbg_suffix}")
        changed = True

    out = "\n".join(out_lines)
    if ir_text.endswith("\n"):
        out += "\n"
    if not changed:
        return out

    out = re.sub(
        r"(?m)^[ \t]*declare\s+i1\s+@llvm\.is\.fpclass\.f32\s*"
        r"\(\s*float\s*,\s*i32\s+immarg\s*\)\s*(?:#\d+)?\s*\n?",
        "",
        out,
    )
    return out


def _rewrite_lifetime_intrinsics_for_llvm14(ir_text: str) -> str:
    out = ir_text

    out = re.sub(
        r"(?m)^([ \t]*(?:tail\s+|musttail\s+|notail\s+)?call\s+void\s+@llvm\.lifetime\.(start|end)\.p0)"
        r"\(\s*ptr(\s+[^()%,]+(?:\s+[^()%,]+)*)?\s+([^,)]+)\s*\)",
        r"\1(i64 -1, ptr\3 \4)",
        out,
    )

    out = re.sub(
        r"(?m)^([ \t]*declare\s+void\s+@llvm\.lifetime\.(start|end)\.p0)"
        r"\(\s*ptr(\s+[^()%,]+(?:\s+[^()%,]+)*)?\s*\)",
        r"\1(i64 immarg, ptr\3)",
        out,
    )

    return out


_SCMP_UCMP_CALL_RE = re.compile(r"^(\s*)(%\w+)\s*=\s*(?:tail\s+|musttail\s+|notail\s+)?call\s+(?P<ret>i\d+)\s+"
                                r"@llvm\.(?P<kind>scmp|ucmp)\.(?P<opty>i\d+)\.(?P=opty)\s*"
                                r"\(\s*(?P=opty)\s+(?P<a>[^,]+)\s*,\s*(?P=opty)\s+(?P<b>[^)]+)\)\s*(?P<tail>.*)$")


def _rewrite_llvm_scmp_ucmp_to_icmp(ir_text: str) -> str:
    pred = {"scmp": ("slt", "sgt"), "ucmp": ("ult", "ugt")}
    out_lines: list[str] = []
    counter = 0
    for line in ir_text.splitlines():
        m = _SCMP_UCMP_CALL_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        counter += 1
        indent = m.group(1)
        result = m.group(2)
        ret_ty = m.group("ret")
        kind = m.group("kind")
        opty = m.group("opty")
        a = m.group("a").strip()
        b = m.group("b").strip()
        tail = m.group("tail").rstrip()
        p_lo, p_hi = pred[kind]
        lt = f"%.musa_scmp_lt_{counter}"
        gt = f"%.musa_scmp_gt_{counter}"
        mid = f"%.musa_scmp_mid_{counter}"
        out_lines.append(f"{indent}{lt} = icmp {p_lo} {opty} {a}, {b}")
        out_lines.append(f"{indent}{gt} = icmp {p_hi} {opty} {a}, {b}")
        out_lines.append(f"{indent}{mid} = select i1 {gt}, {ret_ty} 1, {ret_ty} 0")
        last = f"{indent}{result} = select i1 {lt}, {ret_ty} -1, {ret_ty} {mid}"
        if tail:
            last = f"{last} {tail}"
        out_lines.append(last)

    out = "\n".join(out_lines)
    if ir_text.endswith("\n"):
        out += "\n"
    out = re.sub(
        r"(?m)^[ \t]*declare\s+i\d+\s+@llvm\.(?:scmp|ucmp)\.i\d+\.i\d+\s*"
        r"\(\s*i\d+\s*,\s*i\d+\s*\)[^\n]*\n",
        "",
        out,
    )
    return out


def _llvm_compat(ir_text: str) -> str:
    replacements = [
        ("memory\\(none\\)", "readnone"),
        ("memory\\(read\\)", "readonly"),
        ("memory\\(write\\)", "writeonly"),
        ("memory\\(argmem: readwrite\\)", "argmemonly"),
        ("memory\\(argmem: read\\)", "argmemonly readonly"),
        ("memory\\(argmem: write\\)", "argmemonly writeonly"),
        ("memory\\(inaccessiblemem: readwrite\\)", "inaccessiblememonly"),
        ("memory\\(inaccessiblemem: read\\)", "inaccessiblememonly readonly"),
        ("memory\\(inaccessiblemem: write\\)", "inaccessiblememonly writeonly"),
        ("memory\\(argmem: readwrite, inaccessiblemem: readwrite\\)", "inaccessiblemem_or_argmemonly"),
        ("memory\\(argmem: read, inaccessiblemem: read\\)", "inaccessiblemem_or_argmemonly readonly"),
        ("memory\\(argmem: write, inaccessiblemem: write\\)", "inaccessiblemem_or_argmemonly writeonly"),
    ]
    out = ir_text
    for new, old in replacements:
        out = re.sub(new, old, out)

    out = re.sub(r"\bicmp\s+samesign\b", "icmp", out)

    splat_re = re.compile(r"<(\d+)\s+x\s+([^>]+)>\s+splat\s*\(\s*\2\s+([^)]+)\)")

    def _expand_splat(match: re.Match) -> str:
        count = int(match.group(1))
        ty = match.group(2)
        val = match.group(3)
        elems = ", ".join([f"{ty} {val}"] * count)
        return f"<{count} x {ty}> <{elems}>"

    out = splat_re.sub(_expand_splat, out)
    out = _rewrite_bare_splat_operands(out)
    out = _strip_range_attributes(out)
    out = re.sub(r"\s+captures\(\s*none\s*\)", " nocapture", out)
    out = re.sub(r"\s+captures\([^)]*\)", "", out)
    out = re.sub(r"\bor\s+disjoint\s+", "or ", out)
    out = re.sub(r"\bzext\s+nneg\s+", "zext ", out)
    out = re.sub(r"\bsext\s+nneg\s+", "sext ", out)
    out = re.sub(r"\buitofp\s+nneg\s+", "uitofp ", out)
    out = re.sub(r"\bsitofp\s+nneg\s+", "sitofp ", out)
    out = re.sub(r"\btrunc\s+nuw\s+nsw\s+", "trunc ", out)
    out = re.sub(r"\btrunc\s+nsw\s+nuw\s+", "trunc ", out)
    out = re.sub(r"\btrunc\s+nuw\s+", "trunc ", out)
    out = re.sub(r"\btrunc\s+nsw\s+", "trunc ", out)
    out = re.sub(r"\bgetelementptr\s+inbounds\s+nusw\s+", "getelementptr inbounds ", out)
    out = re.sub(r"\bgetelementptr\s+inbounds\s+nuw\s+", "getelementptr inbounds ", out)
    out = re.sub(r"\bgetelementptr\s+inbounds\s+nsw\s+", "getelementptr inbounds ", out)
    out = re.sub(r"\bgetelementptr\s+nusw\s+", "getelementptr ", out)
    out = re.sub(r"\bgetelementptr\s+nuw\s+", "getelementptr ", out)
    out = re.sub(r"\bgetelementptr\s+nsw\s+", "getelementptr ", out)
    out = _rewrite_musa_isspacep_shared(out)
    out = _rewrite_musa_ptr_gen_to_addrspace(out)
    out = _rewrite_llvm_is_fpclass_f32(out)
    out = _rewrite_lifetime_intrinsics_for_llvm14(out)
    for attr in ("nocallback", "nocreateundeforpoison", "mustprogress", "speculatable", "willreturn"):
        out = re.sub(rf"(?<![A-Za-z0-9_.]){attr}(?![A-Za-z0-9_.])", "", out)
    out = re.sub(r"\bmemory\([^)]*\)", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out


def _extract_kernel_name(ir_text: str) -> str:
    for line in ir_text.splitlines():
        if "nvvm.annotations" in line and "\"kernel\"" in line and "@" in line:
            m = re.search(r"@([A-Za-z_][A-Za-z0-9_\\.]+)", line)
            if m:
                return m.group(1)

    matches = re.findall(r"^define\s+[^@]*@([A-Za-z_][A-Za-z0-9_\.]*)", ir_text, flags=re.MULTILINE)
    if matches:
        return matches[0]
    raise RuntimeError("Unable to determine kernel name from LLVM IR")


def _llc_extra_options(metadata: Dict[str, object], options: "MUSAOptions") -> list[str]:
    uses_mulhi = bool(metadata.get("uses_mulhi_helper"))
    const_calc_opt = [] if uses_mulhi else ["-mtgpu-enable-const-calc=1"]

    uses_sqmma = bool(metadata.get("uses_sqmma"))
    enable_backend_opt = bool(options.enable_llc_opt or options.enable_backend_opt)
    llc_options_map = {
        (False, False): [*const_calc_opt],
        (True, False): {
            *const_calc_opt,
            "-mtgpu-alloc-shared-memory-from-zero=1",
        },
        (False, True): [
            "-mtgpu-enable-const-calc=1",
            "-mtgpu-tiny-offset-hint=1",
            "-mtgpu-combine-instr-with-burst=1",
            "-mtgpu-combine-fop-instr=1",
        ],
        (True, True): [
            "-mtgpu-opt-level=1",
            "-mtgpu-combine-instr-with-burst=1",
            "-mtgpu-combine-fop-instr=1",
            "-misched=mtgpu-max-ilp",
        ],
    }
    opts = llc_options_map[(uses_sqmma, enable_backend_opt)]
    if options.llc_options:
        opts.extend(shlex.split(options.llc_options))
    return opts


@dataclass(frozen=True)
class MUSAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    maxnreg: Optional[int] = None
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    supported_fp8_dtypes: Tuple[str, ...] = ("fp8e5", )
    supported_fp8_storage_dtypes: Tuple[str, ...] = ("fp8e5", )
    custom_fp8_dtypes: Tuple[str, ...] = ()
    deprecated_fp8_dot_operand_dtypes: Tuple[str, ...] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str, ...] = ("ieee", "tf32", "tf32x3", "bf16x3", "bf16x6")
    max_num_imprecise_acc_default: int = 0
    sanitize_overflow: bool = True
    toolchain_path: Optional[str] = None
    llc_path: Optional[str] = None
    lld_path: Optional[str] = None
    llc_asm_path: Optional[str] = None
    llc_options: Optional[str] = None
    enable_llc_opt: bool = False
    enable_backend_opt: bool = False
    enable_fp8_burst2: bool = False
    enable_llvm_compat: bool = True
    extern_libs: Optional[tuple] = None
    debug: bool = False
    backend_name: str = "musa"
    supports_noinline: bool = True
    arch: Optional[str] = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        default_libdir = Path(__file__).parent / "lib"
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get("libdevice", None):
            extern_libs["libdevice"] = knobs.musa.libdevice_path or str(default_libdir / "libdevice.31.bc")
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
            "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        llc_path, lld_path, llc_asm_path = _resolve_toolchain_paths(self)
        hash_dict["effective_llc_path"] = llc_path
        hash_dict["effective_lld_path"] = lld_path
        hash_dict["effective_llc_asm_path"] = llc_asm_path or ""
        hash_dict["effective_llc_major"] = _detect_llvm_major_version(llc_path)
        hash_dict["llc_tool_signature"] = _tool_version_signature(llc_path)
        hash_dict["lld_tool_signature"] = _tool_version_signature(lld_path)
        hash_dict["llc_asm_tool_signature"] = _tool_version_signature(llc_asm_path or "")
        if hash_dict["extern_libs"]:
            hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class MUSABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "musa"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "mubin"

    def parse_options(self, opts) -> Any:
        opts = dict(opts)
        arch = knobs.runtime.override_arch or opts.get("arch", None) or self.target.arch
        args = {"arch": _normalize_arch(arch)}
        capability = _capability_from_arch(args["arch"])
        if opts.get("num_ctas", 1) > 1 and capability == 31:
            raise ValueError("num_ctas > 1 requires MUSA cluster launch support. "
                             f"Current target is {args['arch']} (capability {capability}).")
        if "enable_fp_fusion" not in opts:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion
        if "supported_fp8_dtypes" not in opts:
            supported_fp8_dtypes = {"fp8e5"}
            if capability >= 31:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))
        if "supported_fp8_storage_dtypes" not in opts:
            supported_fp8_storage_dtypes = set(args.get("supported_fp8_dtypes", ()))
            if capability >= 31:
                supported_fp8_storage_dtypes.update({"fp8e4b15", "fp8e4b8", "fp8e5b16"})
            args["supported_fp8_storage_dtypes"] = tuple(sorted(supported_fp8_storage_dtypes))
        if "custom_fp8_dtypes" not in opts:
            custom_fp8_dtypes = set()
            if capability >= 31:
                custom_fp8_dtypes.update({"fp8e4b15", "fp8e4b8", "fp8e5b16"})
            args["custom_fp8_dtypes"] = tuple(sorted(custom_fp8_dtypes))
        if "deprecated_fp8_dot_operand_dtypes" not in opts:
            args["deprecated_fp8_dot_operand_dtypes"] = ()
        if "toolchain_path" not in opts:
            toolchain_path = knobs.musa.toolchain_path
            if not toolchain_path:
                mtcc_bin_path = os.getenv("MTCC_BIN_PATH")
                if mtcc_bin_path:
                    toolchain_path = mtcc_bin_path
                else:
                    musa_home = os.getenv("MUSA_HOME")
                    toolchain_path = str(Path(musa_home) / "bin") if musa_home else None
            args["toolchain_path"] = _normalize_path(toolchain_path)
        if "llc_path" not in opts:
            args["llc_path"] = _select_tool_path(knobs.musa.llc_path, knobs.musa.llc)
        if "lld_path" not in opts:
            args["lld_path"] = _select_tool_path(knobs.musa.lld_path, knobs.musa.lld)
        if "llc_asm_path" not in opts:
            args["llc_asm_path"] = _normalize_path(knobs.musa.llc_asm_path)
        if "llc_options" not in opts:
            args["llc_options"] = knobs.musa.llc_options
        if "enable_llc_opt" not in opts:
            args["enable_llc_opt"] = knobs.musa.enable_llc_opt
        if "enable_fp8_burst2" not in opts:
            args["enable_fp8_burst2"] = knobs.musa.enable_fp8_burst2
        if "enable_llvm_compat" not in opts:
            args["enable_llvm_compat"] = knobs.musa.enable_llvm_compat
        args.update({k: opts[k] for k in MUSAOptions.__dataclass_fields__.keys() if k in opts and opts[k] is not None})
        if "warp_size" not in args:
            target_warp_size = getattr(self.target, "warp_size", None)
            args["warp_size"] = int(target_warp_size) if target_warp_size else 32
        return MUSAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
        )

    def get_codegen_implementation(self, options):
        from triton.language.extra.musa import utils as musa_utils

        return {
            "convert_custom_types": musa_utils.convert_custom_float8,
            "min_dot_size": min_dot_size(self.target),
        }

    def get_module_map(self) -> Dict[str, object]:
        try:
            from triton.language.extra.musa import libdevice as musa_libdevice  # type: ignore
            libdevice = musa_libdevice
        except Exception:
            from triton.language.extra import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        mthreads.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, arch, capability):
        if opt.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        emu_tf32 = capability >= 31

        passes.ttir.add_convert_to_ttgpuir(pm, f"musa:{arch}", opt.num_warps, opt.warp_size, opt.num_ctas)
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, emu_tf32)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)

        mthreads.passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        mthreads.passes.ttgpuir.add_optimize_dot_operands(pm)
        mthreads.passes.ttgpuir.add_optimize_descriptor_encoding(pm)
        passes.ttir.add_loop_aware_cse(pm)

        if capability >= 31:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            mthreads.passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            mthreads.passes.ttgpuir.add_optimize_sqmma_accumulator_layout(pm)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            mthreads.passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        else:
            passes.ttir.add_triton_licm(pm)

        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_prefetch(pm)
        mthreads.passes.ttgpuir.add_optimize_dot_operands(pm)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        mthreads.passes.ttgpuir.add_tme_lowering(pm)
        mthreads.passes.ttgpuir.add_optimize_sqmma_accumulator_layout(pm)
        mthreads.passes.ttgpuir.add_canonicalize_sqmma_result_conversions(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        mthreads.passes.ttgpuir.add_issue_barrier_insertion(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        mthreads.passes.ttgpuir.add_convert_sqmma_to_mtgpu(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        if capability == 31:
            mthreads.passes.ttgpuir.add_mark_inplace_loads(pm)
        mthreads.passes.ttgpuir.add_finalize_barriers(pm)
        pm.run(mod, "make_ttgir")
        metadata["uses_sqmma"] = _module_uses_sqmma(mod)
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_llir(src, metadata, options, arch):
        from triton._C.libtriton import llvm

        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, _capability_from_arch(arch))
        mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, _capability_from_arch(arch))
        mthreads.passes.ttgpuir.add_to_llvmir(pm, _capability_from_arch(arch))
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.convert.add_cf_to_llvmir(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)

        if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
            passes.llvmir.add_di_scope(pm)

        pm.run(mod, "make_llir")

        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        mthreads.attach_datalayout(llvm_mod)

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)
        maxntidx = max(1, int(options.num_warps) * int(options.warp_size))
        kernel_name_hint = src.get_entry_func_name() if hasattr(src, "get_entry_func_name") else ""
        mthreads.decorate_kernel_abi(llvm_mod, kernel_name_hint, maxntidx)
        metadata["uses_mulhi_helper"] = mthreads.module_uses_mulhi_helper(llvm_mod)

        metadata["shared"] = src.get_int_attr("ttg.shared")

        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def make_mubin(src, metadata, opt, arch):
        if not isinstance(src, str):
            raise TypeError("Expected LLVM IR as a string for MUSA codegen")

        llc_path, lld_path, llc_asm_path = _resolve_toolchain_paths(opt)
        if not llc_path or not lld_path:
            raise RuntimeError("MUSA toolchain not configured. Set TRITON_MUSA_TOOLCHAIN_PATH "
                               "or TRITON_MUSA_LLC_PATH/TRITON_MUSA_LLD_PATH (or MUSA_HOME).")

        ir_text = src
        llc_major = _detect_llvm_major_version(llc_path)
        if opt.enable_llvm_compat:
            if _should_apply_llvm_compat(llc_major):
                ir_text = _llvm_compat(ir_text)
        ir_text = _rewrite_llvm_scmp_ucmp_to_icmp(ir_text)

        if knobs.musa.dump_llir:
            print("// -----// MUSA LLVMIR Dump //----- //")
            print(ir_text)

        capability = _capability_from_arch(arch)
        llc_opt_level = "-O2"
        llc_opts = [
            "-march=mtgpu",
            f"-mcpu=mp_{capability}",
            *_llc_opaque_pointer_options(llc_major),
            llc_opt_level,
            "-filetype=obj",
        ]
        llc_opts.extend(_llc_extra_options(metadata, opt))

        tmp_dir = tempfile.mkdtemp(prefix="triton-musa-")
        tmp_path = Path(tmp_dir)
        keep_artifacts = True
        try:
            tmp_path = Path(tmp_dir)
            ll_file = tmp_path / "kernel.ll"
            obj_file = tmp_path / "kernel.o"
            mubin_file = tmp_path / "kernel.mubin"

            ll_file.write_text(ir_text)

            replace_llir = knobs.musa.replace_llir
            if replace_llir and Path(replace_llir).exists():
                ll_file = Path(replace_llir)
            kernel_name = _extract_kernel_name(ll_file.read_text())

            if llc_asm_path:
                llc_asm_major = _detect_llvm_major_version(llc_asm_path)
                asm_file = tmp_path / "kernel.s"
                asm_cmd = [
                    llc_asm_path,
                    str(ll_file),
                    "-march=mtgpu",
                    f"-mcpu=mp_{capability}",
                    *_llc_opaque_pointer_options(llc_asm_major),
                    llc_opt_level,
                    "-filetype=asm",
                    "-o",
                    str(asm_file),
                ]
                asm_cmd.extend(_llc_extra_options(metadata, opt))
                _run_tool_command(
                    "llc-asm",
                    asm_cmd,
                    repro_dir=tmp_path,
                    dump_log=knobs.musa.dump_toolchain_log,
                )
                if knobs.musa.dump_muasm:
                    print("// -----// MUASM Dump //----- //")
                    print(asm_file.read_text())

            llc_cmd = [llc_path, str(ll_file), *llc_opts, "-o", str(obj_file)]
            _run_tool_command(
                "llc",
                llc_cmd,
                repro_dir=tmp_path,
                dump_log=knobs.musa.dump_toolchain_log,
            )

            lld_cmd = [lld_path, "-flavor", "gnu", "-shared", str(obj_file), "-o", str(mubin_file)]
            _run_tool_command(
                "ld.lld",
                lld_cmd,
                repro_dir=tmp_path,
                dump_log=knobs.musa.dump_toolchain_log,
            )

            replace_mubin = knobs.musa.replace_mubin
            if replace_mubin and Path(replace_mubin).exists():
                mubin_file = Path(replace_mubin)

            metadata["name"] = kernel_name
            result = mubin_file.read_bytes()
            keep_artifacts = False
            return result
        finally:
            if not keep_artifacts:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def add_stages(self, stages, options, language):
        arch = options.arch
        capability = _capability_from_arch(arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, arch, capability)
        elif language == Language.GLUON:
            raise RuntimeError("MUSA backend does not support GLUON yet")
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, arch)
        stages["mubin"] = lambda src, metadata: self.make_mubin(src, metadata, options, arch)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, arch)

    @functools.lru_cache()
    def hash(self):
        version = get_musa_version()
        return f"{version}-{self.target.arch}"
