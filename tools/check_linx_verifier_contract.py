#!/usr/bin/env python3
"""Check that decoded representations cannot bypass Linx verification."""

from __future__ import annotations

import re
from pathlib import Path


REPRESENTATION_MARKERS = ("MemRefType", "BindTileOp")


def _contains_representation_marker(text: str) -> bool:
    return any(marker in text for marker in REPRESENTATION_MARKERS)


def _find_matching_open_paren(source: str, close_index: int) -> int | None:
    depth = 0
    for index in range(close_index, -1, -1):
        if source[index] == ")":
            depth += 1
        elif source[index] == "(":
            depth -= 1
            if depth == 0:
                return index
    return None


def _call_argument_count(source: str, open_index: int) -> int | None:
    depth = 0
    commas = 0
    has_content = False
    for character in source[open_index:]:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return commas + 1 if has_content else 0
        elif depth == 1:
            if character == ",":
                commas += 1
            elif not character.isspace():
                has_content = True
    return None


def _has_implicit_linx_fail_closed_dispatch(function: str) -> bool:
    dispatcher = "dispatchVerifierByArch"
    position = 0
    while (position := function.find(dispatcher, position)) >= 0:
        open_index = function.find("(", position + len(dispatcher))
        if open_index < 0:
            return False
        # The overload with two verifier callbacks (three total arguments)
        # synthesizes the fail-closed Linx verifier.
        if _call_argument_count(function, open_index) == 3:
            return True
        position = open_index + 1
    return False


def _has_linx_fail_closed_path(source: str, position: int) -> bool:
    function_start = source.rfind("LogicalResult", 0, position)
    function_end = source.find("\n}\n", position)
    if function_start < 0 or function_end < 0:
        return False
    function = source[function_start:function_end]
    return (
        "VerifierTargetArch::Linx" in function
        or "Linx legality is not implemented" in function
        or _has_implicit_linx_fail_closed_dispatch(function)
    )


def find_independent_early_success_bypasses(source: str) -> list[str]:
    """Find operation-local representation predicates that return success."""
    findings: list[str] = []
    lambda_pattern = re.compile(
        r"auto\s+(?P<name>[A-Za-z_]\w*)\s*=\s*\[[^\]]*\]\s*"
        r"\([^)]*\)\s*(?:->\s*bool\s*)?\{(?P<body>.*?)\n\s*\};",
        re.DOTALL,
    )
    for match in lambda_pattern.finditer(source):
        if not _contains_representation_marker(match.group("body")) or not (
            _has_linx_fail_closed_path(source, match.start())
        ):
            continue
        function_tail = source[match.end() :]
        function_end = function_tail.find("\n}\n")
        if function_end >= 0:
            function_tail = function_tail[:function_end]
        predicate = re.escape(match.group("name"))
        early_success = re.compile(
            rf"if\s*\(\s*{predicate}\s*\(\s*\)\s*\)\s*"
            r"(?:\{\s*)?return\s+(?:mlir::)?success\(\);",
            re.DOTALL,
        )
        if early_success.search(function_tail):
            findings.append(match.group("name"))

    success_pattern = re.compile(r"return\s+(?:mlir::)?success\(\);")
    for match in success_pattern.finditer(source):
        cursor = match.start() - 1
        while cursor >= 0 and source[cursor].isspace():
            cursor -= 1
        if cursor >= 0 and source[cursor] == "{":
            cursor -= 1
            while cursor >= 0 and source[cursor].isspace():
                cursor -= 1
        if cursor < 0 or source[cursor] != ")":
            continue
        open_index = _find_matching_open_paren(source, cursor)
        if open_index is None:
            continue
        keyword_end = open_index
        keyword_start = keyword_end - 1
        while keyword_start >= 0 and source[keyword_start].isspace():
            keyword_start -= 1
        if source[max(0, keyword_start - 1) : keyword_start + 1] != "if":
            continue
        condition = source[open_index + 1 : cursor]
        if _contains_representation_marker(condition) and (
            _has_linx_fail_closed_path(source, open_index)
        ):
            line = source.count("\n", 0, open_index) + 1
            findings.append(f"direct@{line}")
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = (root / "lib/PTO/IR/PTO.cpp").read_text()
    helper_start = source.index("static bool shouldBypassDecodedMemrefVerifier")
    helper_end = source.index("\n}\n", helper_start)
    helper = source[helper_start:helper_end]
    required = "if (isVerifierTargetLinx(op))\n    return false;"
    if required not in helper:
        raise SystemExit(
            "decoded memref/bind_tile verifier bypass must be disabled for Linx"
        )
    independent = find_independent_early_success_bypasses(source)
    if independent:
        raise SystemExit(
            "independent representation-triggered early-success bypasses: "
            + ", ".join(independent)
        )
    print("Linx decoded-representation verifier contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
