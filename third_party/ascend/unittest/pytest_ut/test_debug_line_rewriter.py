# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""Unit tests for debug_line_rewriter.

No real .npubin is needed: minimal ELF + DWARF-4 line programs are synthesized
in memory, so the full parse -> plan -> patch -> verify path is exercised, and
each demotion rule is tested directly against synthetic rows.
"""

import os
import struct
import sys
import tempfile
import unittest
from unittest import mock


def _import_rewriter():
    """Import debug_line_rewriter, preferring the in-tree source so the unit
    test validates the repo code regardless of whether triton-ascend has been
    (re)installed. Falls back to the installed backend if the source isn't
    found next to the test."""
    import importlib

    here = os.path.dirname(os.path.abspath(__file__))
    # pytest_ut/ -> ../../backend/debug_line_rewriter.py ; "." covers in-place runs
    for rel in ("../../backend", "."):
        candidate = os.path.normpath(os.path.join(here, rel))
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        return importlib.import_module("debug_line_rewriter")
    except Exception:
        return importlib.import_module("triton.backends.ascend.debug_line_rewriter")


dlr = _import_rewriter()
import logging as _logging

dlr.log.setLevel(_logging.CRITICAL)  # silence rewrite-skipped logs during tests

rewrite_debug_line = dlr.rewrite_debug_line
rewrite_debug_line_blob = dlr.rewrite_debug_line_blob
ENV_FLAG = dlr.ENV_FLAG
DW_LNS_ADVANCE_LINE = dlr.DW_LNS_ADVANCE_LINE
DW_LNS_COPY = dlr.DW_LNS_COPY
DW_LNS_FIXED_ADVANCE_PC = dlr.DW_LNS_FIXED_ADVANCE_PC
DW_LNS_NEGATE_STMT = dlr.DW_LNS_NEGATE_STMT
DW_LNS_SET_BASIC_BLOCK = dlr.DW_LNS_SET_BASIC_BLOCK
DW_LNS_SET_COLUMN = dlr.DW_LNS_SET_COLUMN
DW_LNS_SET_FILE = dlr.DW_LNS_SET_FILE
DW_LNE_END_SEQUENCE = dlr.DW_LNE_END_SEQUENCE
DW_LNE_SET_ADDRESS = dlr.DW_LNE_SET_ADDRESS

# pytest_ut/conftest.py declares an autouse, module-scoped `assign_npu` fixture
# that depends on `worker_id` (pytest-xdist) and binds an NPU device. This suite
# needs no NPU, so override it with a no-op for this module: the file then runs
# under a bare `pytest <file>` (no xdist) and in NPU-less CI, while the normal
# xdist harness and `python -m unittest` are unaffected. Guarded so unittest
# runs even when pytest is not installed.
try:
    import pytest as _pytest

    @_pytest.fixture(scope="module", autouse=True)
    def assign_npu():  # noqa: D401 - shadows conftest fixture of the same name
        yield
except ImportError:
    pass

# ── builders ──────────────────────────────────────────────────────────────────


def enc_uleb(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def enc_sleb(n):
    out = bytearray()
    more = True
    while more:
        b = n & 0x7F
        n >>= 7
        if (n == 0 and not b & 0x40) or (n == -1 and b & 0x40):
            more = False
        else:
            b |= 0x80
        out.append(b)
    return bytes(out)


def build_debug_line(rows, files=("user_test.py", "internal")):
    """Build a DWARF-4 .debug_line section mimicking bishengir's encoding:
    one set_address, then per row set_file/set_column/negate_stmt/advance_line/
    fixed_advance_pc/copy. `rows` is a list of dicts {line,file,col,is_stmt}."""
    SOL = bytes([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1])  # std opcode arg counts 1..12
    pro = bytearray()
    pro += bytes([1])  # minimum_instruction_length
    pro += bytes([1])  # maximum_operations_per_instruction
    pro += bytes([1])  # default_is_stmt
    pro += struct.pack("b", -5)  # line_base
    pro += bytes([14])  # line_range
    pro += bytes([13])  # opcode_base
    pro += SOL
    pro += bytes([0])  # directory table terminator (empty)
    for name in files:
        pro += name.encode() + b"\0" + enc_uleb(0) + enc_uleb(0) + enc_uleb(0)
    pro += bytes([0])  # file table terminator

    prog = bytearray()
    prog += bytes([0]) + enc_uleb(1 + 8) + bytes([DW_LNE_SET_ADDRESS]) + struct.pack("<Q", 0)
    cur = dict(file=1, line=1, col=0, is_stmt=True)
    for r in rows:
        if r["file"] != cur["file"]:
            prog += bytes([DW_LNS_SET_FILE]) + enc_uleb(r["file"])
            cur["file"] = r["file"]
        if r["col"] != cur["col"]:
            prog += bytes([DW_LNS_SET_COLUMN]) + enc_uleb(r["col"])
            cur["col"] = r["col"]
        if r["is_stmt"] != cur["is_stmt"]:
            prog += bytes([DW_LNS_NEGATE_STMT])
            cur["is_stmt"] = r["is_stmt"]
        if r["line"] != cur["line"]:
            prog += bytes([DW_LNS_ADVANCE_LINE]) + enc_sleb(r["line"] - cur["line"])
            cur["line"] = r["line"]
        prog += bytes([DW_LNS_FIXED_ADVANCE_PC]) + struct.pack("<H", 4)
        prog += bytes([DW_LNS_COPY])
    prog += bytes([0]) + enc_uleb(1) + bytes([DW_LNE_END_SEQUENCE])

    header_length = len(pro)
    body = struct.pack("<H", 4) + struct.pack("<I", header_length) + bytes(pro) + bytes(prog)
    return struct.pack("<I", len(body)) + body


def build_elf(debug_line):
    """Minimal little-endian ELF64 (ET_REL, AARCH64) carrying .debug_line."""
    SHT_PROGBITS, SHT_STRTAB = 1, 3
    names = ["", ".debug_line", ".shstrtab"]
    shstrtab = bytearray(b"\0")
    name_off = {"": 0}
    for n in names[1:]:
        name_off[n] = len(shstrtab)
        shstrtab += n.encode() + b"\0"
    shstrtab = bytes(shstrtab)

    ehsize = shentsize = 64
    dl_off = ehsize
    str_off = dl_off + len(debug_line)
    shoff = str_off + len(shstrtab)

    e_ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    ehdr = e_ident + struct.pack("<HHIQQQIHHHHHH", 1, 183, 1, 0, 0, shoff, 0, ehsize, 0, 0, shentsize, 3, 2)
    assert len(ehdr) == 64

    def shdr(name, stype, offset, size):
        return struct.pack("<IIQQQQIIQQ", name_off[name], stype, 0, 0, offset, size, 0, 0, 1, 0)

    sh0 = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sh1 = shdr(".debug_line", SHT_PROGBITS, dl_off, len(debug_line))
    sh2 = shdr(".shstrtab", SHT_STRTAB, str_off, len(shstrtab))
    return ehdr + debug_line + shstrtab + sh0 + sh1 + sh2


def make_row(line, file=1, is_stmt=True, end=False, kind="copy"):
    return dict(line=line, file=file, col=0, is_stmt=is_stmt, end_sequence=end, emit_kind=kind)


SRC_12_LINES_LOOP_AT_10 = "\n".join(
    ["# 1", "# 2", "# 3", "# 4", "# 5", "# 6", "# 7", "# 8", "# 9", "for i in range(4):", "    pass", "# 12"])

# ── LEB128 ─────────────────────────────────────────────────────────────────────


class TestLEB128(unittest.TestCase):

    def test_uleb_roundtrip(self):
        for n in (0, 1, 63, 64, 127, 128, 300, 16384, 99999):
            v, off = dlr._uleb(enc_uleb(n), 0)
            self.assertEqual(v, n)
            self.assertEqual(off, len(enc_uleb(n)))

    def test_sleb_roundtrip(self):
        for n in (0, 1, -1, 63, -63, 64, -64, -300, 300, -16384):
            v, _ = dlr._sleb(enc_sleb(n), 0)
            self.assertEqual(v, n)


# ── source helpers ─────────────────────────────────────────────────────────────


class TestSourceHelpers(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        self.tmp.write(SRC_12_LINES_LOOP_AT_10)
        self.tmp.close()
        self.addCleanup(os.unlink, self.tmp.name)

    def test_loop_headers(self):
        self.assertEqual(dlr._loop_header_lines(self.tmp.name), {10})

    def test_loop_headers_while_and_async(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write("async def g():\n    while True:\n        async for x in y:\n            pass\n")
        f.close()
        self.addCleanup(os.unlink, f.name)
        # while -> line 2, async for -> line 3
        self.assertEqual(dlr._loop_header_lines(f.name), {2, 3})

    def test_source_length(self):
        self.assertEqual(dlr._source_length(self.tmp.name), 12)

    def test_user_file_indices(self):
        base = os.path.basename(self.tmp.name)
        files = ["", base, "internal", "/x/site-packages/triton/standard.py"]
        self.assertEqual(dlr._user_file_indices(files, self.tmp.name), {1})

    def test_auto_detect_source_prefers_user(self):
        files = ["", "/opt/py/site-packages/triton/standard.py", self.tmp.name]
        self.assertEqual(dlr._auto_detect_source(files), self.tmp.name)

    def test_auto_detect_source_none(self):
        files = ["", "/opt/py/site-packages/triton/standard.py", "/nope/missing.py"]
        self.assertIsNone(dlr._auto_detect_source(files))


# ── demotion rules (pure, synthetic rows) ──────────────────────────────────────


class TestPlanDemotions(unittest.TestCase):

    def plan(self, rows, protected=frozenset(), user_files={1}, src_lines=None):
        return dlr._plan_demotions([rows], protected, user_files, src_lines)

    def test_scatter_dedup(self):
        rows = [make_row(8), make_row(8), make_row(8)]
        demote, kept, counts = self.plan(rows)
        self.assertEqual(len(demote), 2)
        self.assertEqual(kept, [8])
        self.assertEqual(counts["dup"], 2)

    def test_loop_header_kept_every_occurrence(self):
        rows = [make_row(10), make_row(11), make_row(10), make_row(10)]
        demote, kept, counts = self.plan(rows, protected={10})
        self.assertEqual(demote, [])
        self.assertEqual(kept, [10, 11, 10, 10])

    def test_foreign_file_demoted(self):
        rows = [make_row(5, file=2), make_row(7, file=1)]
        demote, kept, counts = self.plan(rows, user_files={1})
        self.assertEqual(counts["foreign"], 1)
        self.assertEqual(kept, [7])

    def test_line_zero_demoted(self):
        rows = [make_row(0, file=1), make_row(7)]
        demote, kept, counts = self.plan(rows)
        self.assertEqual(counts["line0"], 1)
        self.assertEqual(kept, [7])

    def test_over_source_length_demoted(self):
        rows = [make_row(306), make_row(7)]
        demote, kept, counts = self.plan(rows, src_lines=43)
        self.assertEqual(counts["over"], 1)
        self.assertEqual(kept, [7])

    def test_over_disabled_without_src_lines(self):
        rows = [make_row(306), make_row(7)]
        demote, kept, counts = self.plan(rows, src_lines=None)
        self.assertEqual(counts["over"], 0)
        self.assertEqual(kept, [306, 7])

    def test_is_stmt_zero_ignored(self):
        rows = [make_row(7), make_row(0, file=2, is_stmt=False), make_row(8)]
        demote, kept, counts = self.plan(rows)
        self.assertEqual(kept, [7, 8])
        self.assertEqual(len(demote), 0)

    def test_end_sequence_ignored(self):
        rows = [make_row(7), make_row(0, end=True)]
        demote, kept, counts = self.plan(rows)
        self.assertEqual(kept, [7])

    def test_special_opcode_skipped_not_patched(self):
        # a row that WOULD be demoted (over-len) but is emitted by a special
        # opcode -> survives unpatched, counted as special_skip, NOT in demote.
        rows = [make_row(7), make_row(306, kind="special")]
        demote, kept, counts = self.plan(rows, src_lines=43)
        self.assertEqual(len(demote), 0)
        self.assertEqual(counts["special_skip"], 1)
        self.assertIn(306, kept)

    def test_dedup_is_per_sequence(self):
        # line 8 appears once in each of two sequences -> both kept (first per seq).
        demote, kept, counts = dlr._plan_demotions([[make_row(8)], [make_row(8)]], frozenset(), {1}, None)
        self.assertEqual(kept, [8, 8])
        self.assertEqual(len(demote), 0)


# ── simulate (hand-crafted program) ────────────────────────────────────────────


class TestSimulate(unittest.TestCase):

    def test_rows_recovered(self):
        rows = [
            dict(line=7, file=1, col=10, is_stmt=True),
            dict(line=0, file=2, col=0, is_stmt=False),
            dict(line=8, file=1, col=5, is_stmt=True),
        ]
        dl = build_debug_line(rows)
        hdr = dlr._parse_header(dl, little=True)
        self.assertEqual(hdr["version"], 4)
        self.assertEqual(hdr["opcode_base"], 13)
        self.assertEqual(hdr["file_names"], ["", "user_test.py", "internal"])
        seqs = dlr._simulate(dl, hdr)
        self.assertEqual(len(seqs), 1)
        emitted = [(r["line"], r["file"], r["is_stmt"]) for r in seqs[0] if not r["end_sequence"]]
        self.assertEqual(emitted, [(7, 1, True), (0, 2, False), (8, 1, True)])
        # every emitted row carries an opcode offset and kind
        for r in seqs[0]:
            self.assertIn(r["emit_kind"], ("copy", "end"))


# ── end-to-end blob rewrite ─────────────────────────────────────────────────────


class TestRewriteBlob(unittest.TestCase):

    def setUp(self):
        self.src = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        self.src.write(SRC_12_LINES_LOOP_AT_10)
        self.src.close()
        self.addCleanup(os.unlink, self.src.name)
        base = os.path.basename(self.src.name)

        # scatter (8 x3), loop header (10 x2 protected), phantom (99 over-len),
        # line 0 in user file, foreign file row (5 in file 2).
        self.rows = [
            dict(line=7, file=1, col=10, is_stmt=True),
            dict(line=0, file=2, col=0, is_stmt=False),
            dict(line=8, file=1, col=5, is_stmt=True),
            dict(line=8, file=1, col=7, is_stmt=True),
            dict(line=10, file=1, col=3, is_stmt=True),
            dict(line=11, file=1, col=3, is_stmt=True),
            dict(line=10, file=1, col=3, is_stmt=True),
            dict(line=8, file=1, col=9, is_stmt=True),
            dict(line=99, file=1, col=0, is_stmt=True),
            dict(line=0, file=1, col=0, is_stmt=True),
            dict(line=5, file=2, col=0, is_stmt=True),
        ]
        self.blob = build_elf(build_debug_line(self.rows, files=(base, "internal")))

    def test_plan_and_patch(self):
        new, res = rewrite_debug_line_blob(self.blob, src_path=self.src.name)
        self.assertTrue(res.changed)
        self.assertEqual(res.before, [7, 8, 8, 10, 11, 10, 8, 99, 0, 5])
        self.assertEqual(res.after, [7, 8, 10, 11, 10])
        self.assertEqual(res.demoted, 5)
        self.assertEqual(res.counts["dup"], 2)
        self.assertEqual(res.counts["over"], 1)
        self.assertEqual(res.counts["line0"], 1)
        self.assertEqual(res.counts["foreign"], 1)
        self.assertNotEqual(new, self.blob)

    def test_patched_bytes_reparse_to_kept(self):
        new, res = rewrite_debug_line_blob(self.blob, src_path=self.src.name)
        secs, little = dlr._read_sections(__import__("io").BytesIO(new))
        hdr = dlr._parse_header(secs[".debug_line"]["data"], little)
        survivors = dlr._surviving_is_stmt_lines(secs[".debug_line"]["data"], hdr)
        self.assertEqual(survivors, res.after)

    def test_only_copy_bytes_changed_to_basic_block(self):
        new, _ = rewrite_debug_line_blob(self.blob, src_path=self.src.name)
        diff = [i for i in range(len(self.blob)) if self.blob[i] != new[i]]
        # every changed byte was Copy(0x01) and became set_basic_block(0x07)
        for i in diff:
            self.assertEqual(self.blob[i], DW_LNS_COPY)
            self.assertEqual(new[i], DW_LNS_SET_BASIC_BLOCK)
        self.assertEqual(len(diff), 5)
        self.assertEqual(len(new), len(self.blob))  # length preserving

    def test_idempotent(self):
        once, _ = rewrite_debug_line_blob(self.blob, src_path=self.src.name)
        twice, res2 = rewrite_debug_line_blob(once, src_path=self.src.name)
        self.assertFalse(res2.changed)
        self.assertEqual(twice, once)

    def test_noop_when_clean(self):
        base = os.path.basename(self.src.name)
        rows = [dict(line=7, file=1, col=0, is_stmt=True), dict(line=8, file=1, col=0, is_stmt=True)]
        blob = build_elf(build_debug_line(rows, files=(base, "internal")))
        new, res = rewrite_debug_line_blob(blob, src_path=self.src.name)
        self.assertFalse(res.changed)
        self.assertEqual(new, blob)

    def test_wrong_source_does_not_wipe_all_stops(self):
        # source basename matches no file in the table -> foreign rule disabled,
        # so a wrong --src can't demote every row (only the genuine dup goes).
        rows = [
            dict(line=7, file=1, col=0, is_stmt=True),
            dict(line=8, file=1, col=0, is_stmt=True),
            dict(line=8, file=1, col=2, is_stmt=True)
        ]
        blob = build_elf(build_debug_line(rows, files=("user_test.py", "internal")))
        new, res = rewrite_debug_line_blob(blob, src_path="/nonexistent/other.py")
        self.assertEqual(res.counts["foreign"], 0)
        self.assertEqual(res.after, [7, 8])


# ── pipeline entry (env gate, polymorphism, safety) ────────────────────────────


class TestPipelineEntry(unittest.TestCase):

    def setUp(self):
        self.src = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        self.src.write(SRC_12_LINES_LOOP_AT_10)
        self.src.close()
        self.addCleanup(os.unlink, self.src.name)
        base = os.path.basename(self.src.name)
        self.rows = [
            dict(line=7, file=1, col=10, is_stmt=True),
            dict(line=8, file=1, col=5, is_stmt=True),
            dict(line=8, file=1, col=7, is_stmt=True),  # dup -> demote
        ]
        self.blob = build_elf(build_debug_line(self.rows, files=(base, "internal")))

    def test_disabled_is_identity(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_FLAG, None)
            out = rewrite_debug_line(self.blob, metadata=None)
            self.assertEqual(out, self.blob)

    def test_enabled_patches_bytes(self):
        with mock.patch.dict(os.environ, {ENV_FLAG: "1"}):
            # provide source via metadata so it doesn't depend on cwd
            md = {"src_path": self.src.name}
            out = rewrite_debug_line(self.blob, metadata=md)
            self.assertNotEqual(out, self.blob)
            self.assertIsInstance(out, (bytes, bytearray))

    def test_garbage_blob_safe(self):
        with mock.patch.dict(os.environ, {ENV_FLAG: "1"}):
            junk = b"not an elf at all" * 8
            out = rewrite_debug_line(junk, metadata=None)
            self.assertEqual(out, junk)  # returned unchanged, no exception

    def test_path_artifact_patched_in_place(self):
        f = tempfile.NamedTemporaryFile("wb", suffix=".npubin", delete=False)
        f.write(self.blob)
        f.close()
        self.addCleanup(os.unlink, f.name)
        with mock.patch.dict(os.environ, {ENV_FLAG: "1"}):
            md = {"src_path": self.src.name}
            ret = rewrite_debug_line(f.name, metadata=md)
            self.assertEqual(ret, f.name)
        with open(f.name, "rb") as h:
            patched = h.read()
        self.assertNotEqual(patched, self.blob)
        self.assertEqual(len(patched), len(self.blob))

    def test_env_truthy_variants(self):
        for val, expect_change in [("1", True), ("true", True), ("TRUE", True), ("on", True), ("0", False), ("", False),
                                   ("false", False)]:
            with mock.patch.dict(os.environ, {ENV_FLAG: val}):
                md = {"src_path": self.src.name}
                out = rewrite_debug_line(self.blob, metadata=md)
                if expect_change:
                    self.assertNotEqual(out, self.blob, f"val={val!r}")
                else:
                    self.assertEqual(out, self.blob, f"val={val!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
