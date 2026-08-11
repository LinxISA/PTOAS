#!/usr/bin/env python3
"""Verify that the shared nanobind runtime keeps the CPython ABI unresolved on ELF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptoas-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    cmake_path = args.ptoas_root / "lib" / "Bindings" / "Python" / "CMakeLists.txt"
    source = cmake_path.read_text(encoding="utf-8")

    contract = re.compile(
        r"if\s*\(UNIX\s+AND\s+NOT\s+APPLE\).*?"
        r"target_link_options\s*\(nanobind-mlir\s+PRIVATE\s+"
        r'"LINKER:-z,undefs"\s*\).*?endif\s*\(\)',
        re.DOTALL,
    )
    if not contract.search(source):
        raise SystemExit(
            "error: the ELF nanobind-mlir target must override -z defs with "
            "LINKER:-z,undefs so CPython resolves its ABI symbols at module load"
        )

    print("OK: Python binding ELF link contract is explicit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
