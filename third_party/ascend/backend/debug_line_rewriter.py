# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Post-process the DWARF ``.debug_line`` table of a compiled Ascend kernel so
that ``msdebug`` steps once per user source line.

bishengir emits one ``is_stmt`` row per *static* occurrence of a source line.
Instruction scheduling and inlining duplicate those rows (one statement lowered
to several scattered instruction groups, loop bodies, inlined device-library
frames), so plain ``next`` stepping stops repeatedly on the same line and dives
into compiler-generated code. This module removes the redundant stops while
preserving genuine re-entries (loop back-edges).

It complements the MLIR passes ``CanonicalizeDebugLocationsPass`` and
``DeduplicateDebugNopsPass``: those clean up locations *before* bishengir; this
cleans up the residual scatter bishengir introduces *after* them, at the binary
level.

Mechanism (length-preserving, relocation-safe)
    bishengir encodes the line program as a single relocated
    ``DW_LNE_set_address`` followed by ``DW_LNS_fixed_advance_pc`` per row,
    whose operands are symbol differences patched by ``.rela.debug_line``.
    Regenerating the program would reorder those relocations and corrupt the
    table, so instead each dropped stop has its ``DW_LNS_copy`` opcode (0x01)
    overwritten with ``DW_LNS_set_basic_block`` (0x07) -- both one-byte, zero
    argument. ``set_basic_block`` emits no row, so the stop disappears and its
    address folds into the previous row. No byte shifts, so ``.rela.debug_line``
    is never touched and stays valid.

Demotion rules, applied to ``is_stmt=1`` rows in program order:
    1. Scatter   -- within a sequence keep the first row per ``(file, line)``,
       drop the rest.
    2. Loops     -- keep every occurrence of a ``for``/``while`` header line
       (detected from the source AST), so the back-edge re-stops each iteration.
    3a. Foreign  -- drop rows whose file is not the user source file.
    3b. line==0  -- drop compiler-generated rows.
    3c. Over-len -- drop rows whose line exceeds the source length (safety net
        for binaries built before ``CanonicalizeDebugLocationsPass`` or without
        ``--enable-ms-debug``; normally a no-op).

Rows emitted by a DWARF special opcode (e.g. the line-0 function-entry row)
cannot be edited length-preserving and are left as-is; this is harmless because
execution begins there and ``next`` never returns to it.

Public API
    rewrite_debug_line(artifact, metadata=None, options=None)
        Pipeline entry. No-op unless ``LLVM_EXTRACT_DI_LOCAL_VARIABLES`` is set.
        Accepts the ``bytes`` (or path) produced by the ``npubin`` stage and
        returns the rewritten ``bytes`` (or the same path, patched in place).
        Exception-safe: any failure logs and returns the artifact unchanged.

    rewrite_debug_line_blob(blob, src_path=None) -> (bytes, RewriteResult)
        Pure in-memory transform (no env gate); the unit-testable core.

    rewrite_debug_line_file(path, src_path=None, out_path=None) -> RewriteResult
        File convenience used by the CLI; writes atomically.
"""

from __future__ import annotations

import argparse
import ast
import copy
import io
import logging
import os
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from elftools.elf.elffile import ELFFile
except ImportError:  # surfaced lazily; importing this module must never fail
    ELFFile = None

log = logging.getLogger(__name__)

ENV_FLAG = "LLVM_EXTRACT_DI_LOCAL_VARIABLES"

# ── DWARF line-program opcodes ────────────────────────────────────────────────
DW_LNS_COPY = 0x01
DW_LNS_ADVANCE_PC = 0x02
DW_LNS_ADVANCE_LINE = 0x03
DW_LNS_SET_FILE = 0x04
DW_LNS_SET_COLUMN = 0x05
DW_LNS_NEGATE_STMT = 0x06
DW_LNS_SET_BASIC_BLOCK = 0x07
DW_LNS_CONST_ADD_PC = 0x08
DW_LNS_FIXED_ADVANCE_PC = 0x09
DW_LNS_SET_PROLOGUE_END = 0x0A
DW_LNS_SET_EPILOGUE_BEGIN = 0x0B
DW_LNS_SET_ISA = 0x0C
DW_LNE_END_SEQUENCE = 0x01
DW_LNE_SET_ADDRESS = 0x02
DW_LNE_DEFINE_FILE = 0x03
DW_LNE_SET_DISCRIMINATOR = 0x04


@dataclass
class RewriteResult:
    """Outcome of a rewrite attempt."""
    changed: bool
    demoted: int = 0
    before: List[int] = field(default_factory=list)
    after: List[int] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    reason: str = ""


# ── LEB128 ────────────────────────────────────────────────────────────────────


def _uleb(data: bytes, off: int, limit: Optional[int] = None) -> Tuple[int, int]:
    """Decode ULEB128 without reading past ``limit``."""
    limit = len(data) if limit is None else min(limit, len(data))
    result = shift = 0
    while True:
        if off >= limit:
            raise ValueError(f"truncated ULEB128 at .debug_line+{off:#x} (limit {limit:#x})")
        byte = data[off]
        off += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, off
        if shift > 128:
            raise ValueError(f"overlong ULEB128 ending near .debug_line+{off:#x}")


def _sleb(data: bytes, off: int, limit: Optional[int] = None) -> Tuple[int, int]:
    """Decode SLEB128 without reading past ``limit``."""
    limit = len(data) if limit is None else min(limit, len(data))
    result = shift = 0
    while True:
        if off >= limit:
            raise ValueError(f"truncated SLEB128 at .debug_line+{off:#x} (limit {limit:#x})")
        byte = data[off]
        off += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            if shift < 64 and byte & 0x40:
                result |= -(1 << shift)
            return result, off
        if shift > 128:
            raise ValueError(f"overlong SLEB128 ending near .debug_line+{off:#x}")


# ── ELF access (navigation only; the DWARF resolver is never invoked because it
#    chokes on the .npubin's AArch64 relocations and address_size=0) ───────────


def _read_sections(stream) -> Tuple[Dict[str, dict], bool]:
    if ELFFile is None:
        raise RuntimeError("pyelftools is required (pip install pyelftools)")
    elf = ELFFile(stream)
    little = elf.little_endian
    sections: Dict[str, dict] = {}
    for name in (".debug_line", ".rela.debug_line"):
        sec = elf.get_section_by_name(name)
        if sec is not None:
            sections[name] = {
                "data": sec.data(),
                "offset": sec["sh_offset"],
                "size": sec["sh_size"],
            }
    return sections, little


# ── line-program header ───────────────────────────────────────────────────────


def _require(data: bytes, off: int, size: int, limit: int, what: str) -> None:
    if off < 0 or size < 0 or off + size > min(limit, len(data)):
        raise ValueError(f"truncated {what} at .debug_line+{off:#x}: "
                         f"need {size} byte(s), limit {min(limit, len(data)):#x}")


def _cstring(data: bytes, off: int, limit: int, what: str) -> Tuple[str, int]:
    _require(data, off, 1, limit, what)
    end = data.find(b"\0", off, limit)
    if end < 0:
        raise ValueError(f"unterminated {what} at .debug_line+{off:#x}")
    return data[off:end].decode("utf-8", "replace"), end + 1


def _parse_header(data: bytes, little: bool, unit_offset: int = 0) -> dict:
    """Parse one line-table unit beginning at ``unit_offset``.

    The initial length supplies a hard ``unit_end`` boundary.  Linked Ascend
    kernels commonly contain several line units (user code plus libdevice), so
    replaying until the end of the whole section is incorrect.
    """
    endian = "<" if little else ">"
    off = unit_offset
    _require(data, off, 4, len(data), "initial length")
    unit_length, = struct.unpack_from(endian + "I", data, off)
    off += 4
    is_dwarf64 = unit_length == 0xFFFFFFFF
    if is_dwarf64:
        _require(data, off, 8, len(data), "64-bit unit length")
        unit_length, = struct.unpack_from(endian + "Q", data, off)
        off += 8
    elif unit_length >= 0xFFFFFFF0:
        raise ValueError(f"reserved DWARF initial length {unit_length:#x} at .debug_line+{unit_offset:#x}")

    unit_end = off + unit_length
    if unit_end > len(data):
        raise ValueError(f"line unit at .debug_line+{unit_offset:#x} ends at {unit_end:#x}, "
                         f"past section size {len(data):#x}")

    _require(data, off, 2, unit_end, "DWARF version")
    version, = struct.unpack_from(endian + "H", data, off)
    off += 2
    if version < 2 or version > 5:
        raise ValueError(f"unsupported DWARF line version {version} at .debug_line+{unit_offset:#x}")

    address_size = 8
    segment_selector_size = 0
    if version >= 5:
        _require(data, off, 2, unit_end, "DWARF v5 address sizes")
        address_size = data[off]
        off += 1
        segment_selector_size = data[off]
        off += 1

    length_size = 8 if is_dwarf64 else 4
    _require(data, off, length_size, unit_end, "header length")
    header_length, = struct.unpack_from(endian + ("Q" if is_dwarf64 else "I"), data, off)
    off += length_size
    program_start = off + header_length
    if program_start > unit_end:
        raise ValueError(f"line header at .debug_line+{unit_offset:#x} ends at {program_start:#x}, "
                         f"past unit end {unit_end:#x}")

    _require(data, off, 1, program_start, "minimum_instruction_length")
    min_inst_len = data[off]
    off += 1
    if version >= 4:
        _require(data, off, 1, program_start, "maximum_operations_per_instruction")
        off += 1
    _require(data, off, 4, program_start, "line header fields")
    default_is_stmt = data[off]
    off += 1
    line_base = struct.unpack_from("b", data, off)[0]
    off += 1
    line_range = data[off]
    off += 1
    opcode_base = data[off]
    off += 1
    if line_range == 0:
        raise ValueError(f"zero line_range in unit at .debug_line+{unit_offset:#x}")
    if opcode_base == 0:
        raise ValueError(f"zero opcode_base in unit at .debug_line+{unit_offset:#x}")
    _require(data, off, opcode_base - 1, program_start, "standard opcode lengths")
    standard_opcode_lengths = [0] + [data[off + i] for i in range(opcode_base - 1)]
    off += opcode_base - 1

    include_dirs = [""]
    file_names = [""]
    if version <= 4:
        while True:
            directory, off = _cstring(data, off, program_start, "include directory")
            if not directory:
                break
            include_dirs.append(directory)
        while True:
            filename, off = _cstring(data, off, program_start, "file name")
            if not filename:
                break
            file_names.append(filename)
            _, off = _uleb(data, off, program_start)
            _, off = _uleb(data, off, program_start)
            _, off = _uleb(data, off, program_start)
    # DWARF v5 has format-described directory/file tables.  Their extent is
    # already known from header_length; do not parse them as legacy C strings.

    return {
        "unit_start": unit_offset,
        "unit_end": unit_end,
        "is_dwarf64": is_dwarf64,
        "little": little,
        "version": version,
        "address_size": address_size,
        "segment_selector_size": segment_selector_size,
        "min_inst_len": min_inst_len,
        "default_is_stmt": default_is_stmt,
        "line_base": line_base,
        "line_range": line_range,
        "opcode_base": opcode_base,
        "standard_opcode_lengths": standard_opcode_lengths,
        "include_dirs": include_dirs,
        "file_names": file_names,
        "program_start": program_start,
    }


# ── line-number state machine ─────────────────────────────────────────────────


def _simulate(data: bytes, hdr: dict) -> List[List[dict]]:
    """Replay the line program. Each row records ``emit_off`` (byte offset of the
    opcode that produced it) and ``emit_kind`` ('copy' | 'special' | 'end')."""
    ob = hdr["opcode_base"]
    lb = hdr["line_base"]
    lr = hdr["line_range"]
    mil = hdr["min_inst_len"]
    sol = hdr["standard_opcode_lengths"]
    default_stmt = bool(hdr["default_is_stmt"])
    unit_end = hdr.get("unit_end", len(data))
    little = hdr.get("little", True)
    endian = "<" if little else ">"

    def fresh() -> dict:
        return dict(addr=0, file=1, line=1, col=0, is_stmt=default_stmt, end_sequence=False)

    sequences: List[List[dict]] = []
    current: List[dict] = []
    state = fresh()
    off = hdr["program_start"]

    while off < unit_end:
        op_off = off
        opcode = data[off]
        off += 1

        if opcode == 0:  # extended
            length, off = _uleb(data, off, unit_end)
            if length < 1:
                raise ValueError(f"zero-length extended opcode at .debug_line+{op_off:#x}")
            payload_end = off + length
            _require(data, off, length, unit_end, "extended opcode payload")
            sub = data[off]
            off += 1
            if sub == DW_LNE_END_SEQUENCE:
                row = copy.copy(state)
                row["end_sequence"] = True
                row["emit_off"] = op_off
                row["emit_kind"] = "end"
                current.append(row)
                sequences.append(current)
                current = []
                state = fresh()
            elif sub == DW_LNE_SET_ADDRESS:
                # Trust the extended-op length.  Bisheng may advertise
                # address_size=0 while carrying an 8-byte relocated operand.
                operand_size = payload_end - off
                if operand_size <= 0:
                    raise ValueError(f"empty set_address at .debug_line+{op_off:#x}")
                advertised_size = hdr["address_size"]
                if advertised_size not in (0, operand_size):
                    log.debug("debug-line: unit %#x set_address has %d byte(s), header advertises %d",
                              hdr.get("unit_start", 0), operand_size, advertised_size)
                state["addr"] = int.from_bytes(data[off:payload_end], "little" if little else "big")
            elif sub == DW_LNE_SET_DISCRIMINATOR:
                _, off = _uleb(data, off, payload_end)
            elif sub == DW_LNE_DEFINE_FILE:
                _, off = _cstring(data, off, payload_end, "DW_LNE_define_file name")
                for _ in range(3):
                    _, off = _uleb(data, off, payload_end)
            # Known and unknown extended opcodes both end at the declared
            # payload boundary; any producer padding is skipped here.
            off = payload_end
        elif opcode < ob:  # standard
            if opcode == DW_LNS_COPY:
                row = copy.copy(state)
                row["emit_off"] = op_off
                row["emit_kind"] = "copy"
                current.append(row)
            elif opcode == DW_LNS_ADVANCE_PC:
                operand, off = _uleb(data, off, unit_end)
                state["addr"] += operand * mil
            elif opcode == DW_LNS_ADVANCE_LINE:
                operand, off = _sleb(data, off, unit_end)
                state["line"] += operand
            elif opcode == DW_LNS_SET_FILE:
                state["file"], off = _uleb(data, off, unit_end)
            elif opcode == DW_LNS_SET_COLUMN:
                state["col"], off = _uleb(data, off, unit_end)
            elif opcode == DW_LNS_NEGATE_STMT:
                state["is_stmt"] = not state["is_stmt"]
            elif opcode == DW_LNS_SET_BASIC_BLOCK:
                pass
            elif opcode == DW_LNS_CONST_ADD_PC:
                state["addr"] += ((255 - ob) // lr) * mil
            elif opcode == DW_LNS_FIXED_ADVANCE_PC:
                _require(data, off, 2, unit_end, "fixed_advance_pc operand")
                operand, = struct.unpack_from(endian + "H", data, off)
                off += 2
                state["addr"] += operand
            elif opcode in (DW_LNS_SET_PROLOGUE_END, DW_LNS_SET_EPILOGUE_BEGIN):
                pass
            elif opcode == DW_LNS_SET_ISA:
                _, off = _uleb(data, off, unit_end)
            else:  # unknown standard opcode: skip its ULEB operands
                for _ in range(sol[opcode] if opcode < len(sol) else 0):
                    _, off = _uleb(data, off, unit_end)
        else:  # special
            adj = opcode - ob
            state["line"] += lb + (adj % lr)
            state["addr"] += (adj // lr) * mil
            row = copy.copy(state)
            row["emit_off"] = op_off
            row["emit_kind"] = "special"
            current.append(row)

    if current:
        sequences.append(current)
    return sequences


def _decode_units(data: bytes, little: bool) -> List[Tuple[dict, List[List[dict]]]]:
    """Parse and replay every DWARF line unit in section order."""
    units: List[Tuple[dict, List[List[dict]]]] = []
    off = 0
    while off < len(data):
        # Some linkers leave all-zero padding at the end of debug sections.
        if not any(data[off:]):
            break
        hdr = _parse_header(data, little, off)
        units.append((hdr, _simulate(data, hdr)))
        next_off = hdr["unit_end"]
        if next_off <= off:
            raise ValueError(f"non-advancing line unit at .debug_line+{off:#x}")
        off = next_off
    return units


# ── source analysis ───────────────────────────────────────────────────────────


def _loop_header_lines(src_path: str) -> set:
    with open(src_path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=src_path)
    return {node.lineno for node in ast.walk(tree) if isinstance(node, (ast.For, ast.AsyncFor, ast.While))}


def _source_length(src_path: str) -> int:
    with open(src_path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _auto_detect_source(file_names: List[str]) -> Optional[str]:
    """First user (.py, not site-packages) file from the line table that exists
    on disk, resolved relative to the current working directory."""
    for path in file_names:
        if path.endswith(".py") and "/site-packages/" not in path and os.path.isfile(path):
            return path
    return None


def _user_file_indices(file_names: List[str], src_path: str) -> set:
    base = os.path.basename(src_path)
    return {i for i, p in enumerate(file_names) if os.path.basename(p) == base}


# ── demotion planning ─────────────────────────────────────────────────────────


def _plan_demotions(sequences, protected, user_files, src_lines):
    demote: List[dict] = []
    kept: List[int] = []
    counts = {"dup": 0, "foreign": 0, "line0": 0, "over": 0, "special_skip": 0}
    for sequence in sequences:
        seen = set()
        for row in sequence:
            if row["end_sequence"] or not row["is_stmt"]:
                continue
            reason = None
            if user_files is not None and row["file"] not in user_files:
                reason = "foreign"
            elif row["line"] == 0:
                reason = "line0"
            elif src_lines and row["line"] > src_lines:
                reason = "over"
            elif row["line"] in protected:
                reason = None  # keep every occurrence of a loop header
            else:
                key = (row["file"], row["line"])
                if key in seen:
                    reason = "dup"
                else:
                    seen.add(key)
            if reason and row["emit_kind"] != "special":
                demote.append(row)
                counts[reason] += 1
            else:
                if reason:  # wanted but special -> survives unpatched
                    counts["special_skip"] += 1
                kept.append(row["line"])
    return demote, kept, counts


def _surviving_is_stmt_lines(data: bytes, hdr: dict) -> List[int]:
    return [
        row["line"]
        for sequence in _simulate(data, hdr)
        for row in sequence
        if not row["end_sequence"] and row["is_stmt"]
    ]


# ── core transform (in-memory, no env gate) ──────────────────────────────────


def rewrite_debug_line_blob(blob: bytes, src_path: Optional[str] = None) -> Tuple[bytes, RewriteResult]:
    """Rewrite the ``.debug_line`` table inside an ELF ``blob``.

    Returns ``(new_blob, result)``. ``new_blob`` is the original ``blob`` when
    no change is made or verification fails (the input is never corrupted).
    """
    sections, little = _read_sections(io.BytesIO(blob))
    if ".debug_line" not in sections:
        return blob, RewriteResult(False, reason="no .debug_line section")

    debug_line = sections[".debug_line"]["data"]
    units = _decode_units(debug_line, little)
    if not units:
        return blob, RewriteResult(False, reason="empty .debug_line section")
    log.debug("debug-line: parsed %d unit(s): %s", len(units), [
        f"{hdr['unit_start']:#x}-{hdr['unit_end']:#x}/v{hdr['version']}/addr{hdr['address_size']}" for hdr, _ in units
    ])

    src = src_path
    if not src:
        for hdr, _ in units:
            src = _auto_detect_source(hdr["file_names"])
            if src:
                break
    protected = _loop_header_lines(src) if src and os.path.isfile(src) else set()
    src_lines = _source_length(src) if src and os.path.isfile(src) else None

    before: List[int] = []
    demote: List[dict] = []
    kept: List[int] = []
    counts = {"dup": 0, "foreign": 0, "line0": 0, "over": 0, "special_skip": 0}
    for hdr, sequences in units:
        user_files = _user_file_indices(hdr["file_names"], src) if src else None
        # A source mismatch (or an unresolved v5 file table) must not cause all
        # rows in this unit to be classified as foreign.
        if user_files is not None and not user_files:
            log.warning("debug-line: source %s matches no file in unit %#x table %s; skipping foreign-file rule", src,
                        hdr["unit_start"], hdr["file_names"][1:])
            user_files = None
        before.extend(row["line"]
                      for sequence in sequences
                      for row in sequence
                      if row["is_stmt"] and not row["end_sequence"])
        unit_demote, unit_kept, unit_counts = _plan_demotions(sequences, protected, user_files, src_lines)
        demote.extend(unit_demote)
        kept.extend(unit_kept)
        for key in counts:
            counts[key] += unit_counts[key]

    if not demote:
        return blob, RewriteResult(False, before=before, after=kept, counts=counts, reason="nothing to demote")

    base = sections[".debug_line"]["offset"]
    patched = bytearray(blob)
    for row in demote:
        file_off = base + row["emit_off"]
        if patched[file_off] != DW_LNS_COPY:
            return blob, RewriteResult(
                False, before=before, after=kept, counts=counts,
                reason=f"expected Copy(0x01) at .debug_line+{row['emit_off']:#x}, "
                f"found {patched[file_off]:#04x}")
        patched[file_off] = DW_LNS_SET_BASIC_BLOCK

    # Verify against the patched bytes (length-preserving, so offsets are stable).
    verify_sections, verify_little = _read_sections(io.BytesIO(bytes(patched)))
    verify_debug_line = verify_sections[".debug_line"]["data"]
    survivors = [
        line for verify_hdr, _ in _decode_units(verify_debug_line, verify_little)
        for line in _surviving_is_stmt_lines(verify_debug_line, verify_hdr)
    ]
    if survivors != kept:
        return blob, RewriteResult(False, before=before, after=kept, counts=counts,
                                   reason=f"verify mismatch: survivors={survivors} kept={kept}")

    return bytes(patched), RewriteResult(True, demoted=len(demote), before=before, after=kept, counts=counts,
                                         reason="ok")


# ── file convenience (CLI / manual use) ───────────────────────────────────────


def rewrite_debug_line_file(path: str, src_path: Optional[str] = None, out_path: Optional[str] = None) -> RewriteResult:
    """Rewrite ``path``; write to ``out_path`` (or in place when ``None``).
    Writes atomically via a temp file + ``os.replace``."""
    with open(path, "rb") as handle:
        blob = handle.read()
    new_blob, result = rewrite_debug_line_blob(blob, src_path)
    if not result.changed:
        return result

    dest = out_path or path
    dest_dir = os.path.dirname(os.path.abspath(dest)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".debugline_", suffix=".npubin", dir=dest_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(new_blob)
        os.replace(tmp, dest)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return result


# ── pipeline entry (env-gated, polymorphic, exception-safe) ───────────────────


def _enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_source(metadata) -> Optional[str]:
    """Best-effort source path from compiler metadata; ``None`` falls back to
    auto-detection from the line table's file_names."""
    if metadata is None:
        return None
    for attr in ("src_path", "source_path", "filename"):
        value = getattr(metadata, attr, None)
        if isinstance(value, str) and os.path.isfile(value):
            return value
    if isinstance(metadata, dict):
        for key in ("src_path", "source_path", "filename"):
            value = metadata.get(key)
            if isinstance(value, str) and os.path.isfile(value):
                return value
    return None


def rewrite_debug_line(artifact, metadata=None, options=None):
    """``npubin``-stage post-processor. Returns ``artifact`` unchanged unless
    ``LLVM_EXTRACT_DI_LOCAL_VARIABLES`` is set. Accepts the stage's ``bytes`` (or
    a path) and returns the rewritten ``bytes`` (or the same path, patched in
    place). Never raises: failures are logged and the artifact is returned as-is.
    """
    if not _enabled():
        return artifact

    src = _resolve_source(metadata)
    try:
        if isinstance(artifact, (bytes, bytearray)):
            new_blob, result = rewrite_debug_line_blob(bytes(artifact), src)
            if result.changed:
                log.info("debug-line: demoted %d stop(s) %s -> %s", result.demoted, result.before, result.after)
            else:
                log.debug("debug-line: no change (%s)", result.reason)
            return new_blob
        if isinstance(artifact, str):
            result = rewrite_debug_line_file(artifact, src_path=src)
            if result.changed:
                log.info("debug-line: demoted %d stop(s) in %s", result.demoted, artifact)
            return artifact
        log.warning("debug-line: unexpected npubin artifact type %r; skipping", type(artifact))
        return artifact
    except Exception:  # never break the build over debug-info cleanup
        log.exception("debug-line: rewrite skipped due to error")
        return artifact


# ── CLI ───────────────────────────────────────────────────────────────────────


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite a .npubin's .debug_line for clean msdebug stepping.")
    parser.add_argument("binary")
    parser.add_argument("--src", help="kernel .py (loop detection + user-file id)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--out", help="write patched copy here")
    group.add_argument("--in-place", action="store_true", help="patch in place")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    with open(args.binary, "rb") as handle:
        blob = handle.read()

    if args.dry_run:
        _, result = rewrite_debug_line_blob(blob, args.src)
        log.info("demote %d (dup:%d foreign:%d line0:%d over_src_len:%d special_skipped:%d)", result.demoted,
                 result.counts.get("dup", 0), result.counts.get("foreign", 0), result.counts.get("line0", 0),
                 result.counts.get("over", 0), result.counts.get("special_skip", 0))
        log.info("before: %s", result.before)
        log.info("after:  %s", result.after)
        log.info("(dry run — no write; reason=%s)", result.reason)
        return 0

    out = None if args.in_place else (args.out or args.binary + ".patched.npubin")
    result = rewrite_debug_line_file(args.binary, src_path=args.src, out_path=out)
    if result.changed:
        log.info("demoted %d stop(s): %s -> %s", result.demoted, result.before, result.after)
        log.info("written: %s", out or args.binary)
        if out:
            log.info("install: cp %s %s", out, args.binary)
        return 0
    log.info("no change (%s)", result.reason)
    return 1


if __name__ == "__main__":
    sys.exit(_main())
