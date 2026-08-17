#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate reproducible Docker source and ISA-tag workflow behavior."""

from __future__ import annotations

import argparse
import glob
import shlex
import subprocess
from pathlib import Path


def local_copy_sources(dockerfile: str) -> list[tuple[int, str]]:
    """Return local Docker COPY sources and reject unvalidated syntax."""
    sources: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(dockerfile.splitlines(), start=1):
        line = raw_line.strip()
        if not line.startswith("COPY "):
            continue
        tokens = shlex.split(line)
        if tokens[0] != "COPY":
            continue
        index = 1
        from_stage = False
        while index < len(tokens) and tokens[index].startswith("--"):
            if tokens[index].startswith("--from="):
                from_stage = True
            index += 1
        if from_stage:
            continue
        operands = tokens[index:]
        if len(operands) < 2:
            raise SystemExit(f"Docker COPY on line {line_number} is malformed")
        for source in operands[:-1]:
            sources.append((line_number, source))
    return sources


def validate_local_copy_sources(root: Path, dockerfile: str) -> None:
    root = root.resolve()
    for line_number, source in local_copy_sources(dockerfile):
        if source.startswith("/") or "$" in source:
            raise SystemExit(
                f"Docker COPY source on line {line_number} is not a fixed "
                f"repository-root path: {source}"
            )
        candidate = root / source
        matches = glob.glob(str(candidate))
        if not matches:
            raise SystemExit(
                f"Docker COPY source on line {line_number} does not exist "
                f"from repository-root context: {source}"
            )
        for match in matches:
            try:
                Path(match).resolve().relative_to(root)
            except ValueError as error:
                raise SystemExit(
                    f"Docker COPY source escapes repository root: {source}"
                ) from error


def active_shell_lines(script: str) -> set[str]:
    """Return executable, non-comment shell lines for structural checks."""
    return {
        line.strip()
        for line in script.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def validate_cli_identity_helper(root: Path) -> None:
    """Exercise the packaged CLI identity validator with positive and negative inputs."""
    helper = root / "docker/check_ptoas_cli_identity.sh"

    def run(output: str, product_version: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(helper), output, product_version],
            check=False,
            capture_output=True,
            text=True,
        )

    for output, product_version in (
        ("ptoas 0.41 (PTO ISA 0.58.1)", "0.41"),
        ("ptoas 0.41 (PTO ISA 0.58.1)", ""),
    ):
        result = run(output, product_version)
        if result.returncode != 0:
            raise SystemExit(
                f"packaged CLI identity validator rejected valid input: {output}"
            )

    for output in (
        "ptoas 0.41",
        "ptoas 0.40 (PTO ISA 0.58.1)",
        "ptoas 0.41 (PTO ISA 0.58.0)",
        "warning\nptoas 0.41 (PTO ISA 0.58.1)",
        "ptoas 0.41 (PTO ISA 0.58.1)\nptoas 0.40",
    ):
        if run(output, "0.41").returncode == 0:
            raise SystemExit(
                f"packaged CLI identity validator accepted invalid input: {output!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ptoas-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.ptoas_root.resolve()

    dockerfile = (root / "docker/Dockerfile").read_text()
    validate_local_copy_sources(root, dockerfile)
    if "COPY . ${PTO_SOURCE_DIR}" not in dockerfile:
        raise SystemExit("Docker build must COPY the reviewed PTOAS checkout")
    if "git clone https://github.com/zhangstevenunity/PTOAS.git" in dockerfile:
        raise SystemExit("Docker build must not clone an unpinned PTOAS fork")
    if (
        'test "$(git -C pto-isa rev-parse HEAD)" = "${PTO_ISA_COMMIT}"'
        not in dockerfile
    ):
        raise SystemExit("Docker build must assert the exact PTO ISA checkout")

    readme = (root / "docker/README.md").read_text()
    if "docker build -f docker/Dockerfile ." not in readme:
        raise SystemExit("Docker README must document repository-root context")

    workflow = (root / ".github/workflows/isa_contract.yml").read_text()
    if "linxisa-v0.58.1" not in workflow:
        raise SystemExit("raw linxisa-v0.58.1 tag is not an ISA checker trigger")
    exact_command = "python3 tools/check_v058_pto_manifest.py --ptoas-root ."
    if exact_command not in workflow:
        raise SystemExit("ISA tag workflow does not run the exact contract checker")
    for required in (
        "docker/setup-buildx-action@",
        "docker/build-push-action@",
        "context: .",
        "file: docker/Dockerfile",
        "target: builder",
        "push: false",
    ):
        if required not in workflow:
            raise SystemExit(f"hosted Docker builder gate is missing: {required}")

    for name in ("build_wheel.yml", "build_wheel_mac.yml"):
        text = (root / ".github/workflows" / name).read_text()
        if 'GITHUB_REF_NAME}" = "linxisa-v0.58.1"' in text:
            raise SystemExit(
                f"{name} conflates ISA identity with PTOAS product version"
            )

    validate_cli_identity_helper(root)
    identity_invocation = (
        'bash "${PTO_SOURCE_DIR}/docker/check_ptoas_cli_identity.sh" '
        '"${VERSION_OUTPUT}" "${PTOAS_VERSION:-}"'
    )
    for name in (
        "docker/test_ptoas_cli.sh",
        "docker/collect_ptoas_dist.sh",
        "docker/collect_ptoas_dist_mac.sh",
    ):
        text = (root / name).read_text()
        if identity_invocation not in active_shell_lines(text):
            raise SystemExit(
                f"{name} does not execute the exact PTO ISA 0.58.1 CLI identity validator"
            )

    print("release delivery contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
