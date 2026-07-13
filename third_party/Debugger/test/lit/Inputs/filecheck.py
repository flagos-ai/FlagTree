# SPDX-License-Identifier: MIT
"""Small FileCheck-compatible fallback for the debugger's self-contained tests."""

import re
import sys


def _read_checks(path):
    marker = re.compile(r"^\s*(?://|#|;)\s*(CHECK(?:-[A-Z]+)?):\s*(.*)$")
    with open(path, "r", encoding="utf-8") as stream:
        return [(match.group(1), match.group(2), line_no)
                for line_no, line in enumerate(stream, 1)
                if (match := marker.match(line.rstrip("\n")))]


def _compile_pattern(pattern, variables):
    pieces = []
    definitions = []
    cursor = 0
    while cursor < len(pattern):
        if pattern.startswith("{{", cursor):
            end = pattern.find("}}", cursor + 2)
            if end < 0:
                raise ValueError("unterminated {{ regex }}")
            pieces.append("(" + pattern[cursor + 2:end] + ")")
            cursor = end + 2
            continue

        if pattern.startswith("[[", cursor):
            end = pattern.find("]]", cursor + 2)
            if end < 0:
                raise ValueError("unterminated [[ variable ]]")
            body = pattern[cursor + 2:end]
            if ":" in body:
                name, regex = body.split(":", 1)
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    raise ValueError(f"invalid variable name {name!r}")
                pieces.append(f"(?P<{name}>{regex})")
                definitions.append(name)
            else:
                if body not in variables:
                    raise ValueError(f"undefined variable {body!r}")
                pieces.append(re.escape(variables[body]))
            cursor = end + 2
            continue

        next_markers = [
            position for position in (pattern.find("{{", cursor), pattern.find("[[", cursor)) if position >= 0
        ]
        end = min(next_markers) if next_markers else len(pattern)
        pieces.append(re.escape(pattern[cursor:end]))
        cursor = end

    return re.compile("".join(pieces)), definitions


def _line_bounds(text, position):
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return start, len(text) if end < 0 else end


def _fail(path, line_no, message, pattern=None):
    print(f"{path}:{line_no}: {message}", file=sys.stderr)
    if pattern is not None:
        print(f"  pattern: {pattern}", file=sys.stderr)
    return 1


def main(argv):
    positional = [arg for arg in argv[1:] if not arg.startswith("--check-prefix")]
    unsupported = [arg for arg in positional if arg.startswith("--")]
    if unsupported or len(positional) != 1:
        print("usage: filecheck.py CHECK_FILE", file=sys.stderr)
        return 2

    check_file = positional[0]
    try:
        directives = _read_checks(check_file)
    except OSError as error:
        print(error, file=sys.stderr)
        return 2

    text = sys.stdin.read()
    variables = {}
    cursor = 0
    line_start = 0
    line_end = 0
    line_cursor = 0
    pending_not = []

    for kind, pattern, line_no in directives:
        try:
            regex, definitions = _compile_pattern(pattern, variables)
        except ValueError as error:
            return _fail(check_file, line_no, str(error), pattern)

        if kind == "CHECK-NOT":
            pending_not.append((regex, pattern, line_no))
            continue

        if kind == "CHECK-SAME":
            match = regex.search(text[line_start:line_end], max(0, line_cursor - line_start))
            if match is None:
                return _fail(check_file, line_no, "CHECK-SAME pattern not found on current line", pattern)
            absolute_start = line_start + match.start()
            absolute_end = line_start + match.end()
        elif kind in ("CHECK", "CHECK-LABEL"):
            match = regex.search(text, cursor)
            if match is None:
                return _fail(check_file, line_no, f"{kind} pattern not found", pattern)
            absolute_start, absolute_end = match.start(), match.end()
        else:
            return _fail(check_file, line_no, f"unsupported directive {kind}", pattern)

        for not_regex, not_pattern, not_line in pending_not:
            if not_regex.search(text[cursor:absolute_start]):
                return _fail(check_file, not_line, "CHECK-NOT pattern found before next match", not_pattern)
        pending_not.clear()

        for name in definitions:
            variables[name] = match.group(name)
        line_start, line_end = _line_bounds(text, absolute_start)
        line_cursor = absolute_end
        cursor = absolute_end

    for not_regex, not_pattern, not_line in pending_not:
        if not_regex.search(text[cursor:]):
            return _fail(check_file, not_line, "CHECK-NOT pattern found after last match", not_pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
