#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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
