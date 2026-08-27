#
# Copyright 2024 Enflame. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""GCU intrinsic <-> placeholder mapping for TLE-Raw.
GCU-specific LLVM intrinsics (@llvm.tcle.* / @llvm.gcu.*) cannot be imported
by mlir::translateLLVMIRToModule because no LLVMImportDialectInterface is
registered for them.  We rename them to generic external function calls before
MLIR import and restore the original names before gcu-compiler-compile so
the GCU LLVM backend can emit proper instructions.
The mapping table is **dynamically generated** by parsing the TableGen'd header
files IntrinsicsTCLE_gcu400.txt and IntrinsicsGCU_gcu400.txt at first import.  Each header line:
    tcle_read_bid_x,                           // llvm.tcle.read.bid.x
produces the mapping:
    "llvm.tcle.read.bid.x" -> "tcle_read_bid_x"
Type suffixes (e.g., .i32, .i64) are NOT part of the base intrinsic name.
They are captured dynamically during forward rewriting and encoded into the
placeholder name using a '___' delimiter so reverse restoration can reconstruct
the full intrinsic name.
Example round-trip:
    @llvm.tcle.read.bid.x.i32 --> @tcle_read_bid_x___i32 --> @llvm.tcle.read.bid.x.i32
Usage:
    from triton.backends.enflame.gcu_intrinsics import (
        rewrite_intrinsics_to_placeholders,
        restore_intrinsics_from_placeholders,
    )
"""
import os
import re
from typing import Dict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HEADER_LINE_RE = re.compile(r'^\s+(\w+)(?:\s*=\s*\d+)?\s*,\s*//\s*(llvm\.\w[\w.]*\w)\s*$')


def _find_header_dir() -> str:
    """Locate the directory containing IntrinsicsTCLE_gcu400.txt / IntrinsicsGCU_gcu400.txt."""
    env = os.environ.get("GCU_INTRINSICS_HEADER_DIR")
    if env and os.path.isdir(env):
        return env
    if os.path.isfile(os.path.join(_THIS_DIR, "IntrinsicsTCLE_gcu400.txt")):
        return _THIS_DIR
    # When installed via pip, headers may not be co-located; search source tree.
    src_backend = os.path.join(os.path.dirname(_THIS_DIR), "third_party", "enflame", "backend")
    if os.path.isfile(os.path.join(src_backend, "IntrinsicsTCLE_gcu400.txt")):
        return src_backend
    # Try common flagtree workspace layout.
    for candidate in (
            os.path.expanduser("~/flagtree_checkin/flagtree/third_party/enflame/backend"),
            os.path.expanduser("~/flagtree/third_party/enflame/backend"),
    ):
        if os.path.isfile(os.path.join(candidate, "IntrinsicsTCLE_gcu400.txt")):
            return candidate
    return _THIS_DIR


def _parse_header(path: str) -> Dict[str, str]:
    """Parse a TableGen'd intrinsic enum header into {llvm_name: enum_name}."""
    table: Dict[str, str] = {}
    if not os.path.isfile(path):
        return table
    with open(path, 'r') as f:
        for line in f:
            m = _HEADER_LINE_RE.match(line)
            if m:
                enum_name = m.group(1)  # e.g. tcle_read_bid_x
                llvm_name = m.group(2)  # e.g. llvm.tcle.read.bid.x
                table[llvm_name] = enum_name
    return table


def _build_table() -> Dict[str, str]:
    """Build intrinsic table from IntrinsicsTCLE_gcu400.txt and IntrinsicsGCU_gcu400.txt."""
    header_dir = _find_header_dir()
    table: Dict[str, str] = {}
    for header in ("IntrinsicsTCLE_gcu400.txt", "IntrinsicsGCU_gcu400.txt"):
        path = os.path.join(header_dir, header)
        table.update(_parse_header(path))
    return table


# Lazily built on first access.
_table_cache: Dict[str, str] | None = None


def _get_table() -> Dict[str, str]:
    global _table_cache
    if _table_cache is None:
        _table_cache = _build_table()
    return _table_cache


# Public alias for external inspection / testing.
def get_intrinsic_table() -> Dict[str, str]:
    """Return the full {llvm_base_name: placeholder_name} mapping."""
    return _get_table()


# --- Forward / reverse regex (built lazily) ---
_forward_pattern_cache = None
_reverse_pattern_cache = None
_placeholder_to_base_cache: Dict[str, str] | None = None


def _get_forward_pattern():
    global _forward_pattern_cache
    if _forward_pattern_cache is None:
        table = _get_table()
        _forward_pattern_cache = re.compile(r'@(' +
                                            '|'.join(re.escape(k)
                                                     for k in sorted(table.keys(), key=len, reverse=True)) +
                                            r')\.(\w+)')
    return _forward_pattern_cache


def _get_reverse_pattern():
    global _reverse_pattern_cache, _placeholder_to_base_cache
    if _reverse_pattern_cache is None:
        table = _get_table()
        _placeholder_to_base_cache = {v: k for k, v in table.items()}
        _reverse_pattern_cache = re.compile(r'(' + '|'.join(
            re.escape(v) for v in sorted(table.values(), key=len, reverse=True)) + r')___(\w+)')
    return _reverse_pattern_cache


def rewrite_intrinsics_to_placeholders(llvm_ir: str) -> str:
    """Replace @llvm.tcle.*/gcu.* intrinsics with @<placeholder>___<suffix>.
    Captures the type suffix and encodes it into the placeholder name so
    reverse restoration can reconstruct the full intrinsic name.
    """
    table = _get_table()
    pattern = _get_forward_pattern()

    def _replace(m):
        base = m.group(1)
        suffix = m.group(2)
        placeholder = table[base]
        return f"@{placeholder}___{suffix}"

    return pattern.sub(_replace, llvm_ir)


def restore_intrinsics_from_placeholders(mlir_text: str) -> str:
    """Restore <placeholder>___<suffix> back to llvm.tcle.*/gcu.* names."""
    pattern = _get_reverse_pattern()
    p2b = _placeholder_to_base_cache

    def _restore(m):
        placeholder = m.group(1)
        suffix = m.group(2)
        base = p2b[placeholder]
        return f"{base}.{suffix}"

    return pattern.sub(_restore, mlir_text)
