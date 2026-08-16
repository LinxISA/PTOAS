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


def extract_workflow_step(workflow: str, name: str) -> str | None:
    lines = workflow.splitlines()
    marker = f"- name: {name}"
    for index, line in enumerate(lines):
        if line.strip() != marker:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        end = index + 1
        while end < len(lines) and not lines[end].startswith(f"{indent}- "):
            end += 1
        return "\n".join(lines[index:end])
    return None


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

    install_contract = re.compile(
        r"install\s*\(\s*TARGETS\s+nanobind-mlir\s+"
        r"LIBRARY\s+DESTINATION\s+lib\s+"
        r"(?:COMPONENT\s+PTOASPythonRuntime\s+)?\)",
        re.DOTALL,
    )
    if not install_contract.search(source):
        raise SystemExit(
            "error: install nanobind-mlir into lib so auditwheel can locate "
            "the NB_SHARED runtime"
        )

    workflow_path = args.ptoas_root / ".github" / "workflows" / "build_wheel.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    repair_step = extract_workflow_step(workflow, "Repair wheel with auditwheel")
    if repair_step is None:
        raise SystemExit(
            "error: workflow is missing the Repair wheel with auditwheel step"
        )
    active_repair_lines = "\n".join(
        line for line in repair_step.splitlines() if not line.lstrip().startswith("#")
    )
    search_path = re.search(
        r"(?m)^\s*export\s+LD_LIBRARY_PATH=(?P<value>[^\n#]+)$",
        active_repair_lines,
    )
    if search_path is None or not re.search(
        r"\$\{?PTO_INSTALL_DIR\}?/lib", search_path.group("value")
    ):
        raise SystemExit(
            "error: Repair wheel with auditwheel must export LD_LIBRARY_PATH "
            "including $PTO_INSTALL_DIR/lib"
        )
    if not re.search(
        r"(?m)^\s*auditwheel\s+repair(?:\s|$)", active_repair_lines
    ):
        raise SystemExit(
            "error: Repair wheel with auditwheel must invoke auditwheel repair "
            "in the same step that exports LD_LIBRARY_PATH"
        )

    print(
        "OK: Python binding ELF link and shared-runtime packaging contract is explicit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
