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


def run_packaging_identity_mode(
    script: Path, output: str, product_version: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run one packaging script's executable identity-only path."""
    return subprocess.run(
        [
            "bash",
            str(script),
            "--check-ptoas-cli-identity",
            output,
            product_version,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def validate_packaging_identity_mode(script: Path) -> None:
    """Exercise a packaging script with valid and adversarial identity outputs."""

    for output, product_version in (
        ("ptoas 0.41 (PTO ISA 0.58.3)", "0.41"),
        ("ptoas 0.41 (PTO ISA 0.58.3)", ""),
    ):
        result = run_packaging_identity_mode(script, output, product_version)
        if result.returncode != 0:
            raise SystemExit(
                f"{script.name} rejected valid packaged CLI identity: {output}"
            )

    invalid_outputs = (
        "ptoas 0.41",
        "ptoas 0.40 (PTO ISA 0.58.3)",
        "ptoas 0.41 (PTO ISA 0.58.0)",
        "warning\nptoas 0.41 (PTO ISA 0.58.3)",
        "ptoas 0.41 (PTO ISA 0.58.3)\nptoas 0.40",
    )
    for product_version in ("0.41", ""):
        for output in invalid_outputs:
            if (
                run_packaging_identity_mode(script, output, product_version).returncode
                == 0
            ):
                raise SystemExit(
                    f"{script.name} accepted invalid packaged CLI identity "
                    f"in product mode {product_version!r}: {output!r}"
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
    if "'nanobind>=2.9,<3'" not in dockerfile:
        raise SystemExit(
            "Docker LLVM source build must pin nanobind to the MLIR 23 "
            "compatible >=2.9,<3 range"
        )
    if (
        'test "$(git -C pto-isa rev-parse HEAD)" = "${PTO_ISA_COMMIT}"'
        not in dockerfile
    ):
        raise SystemExit("Docker build must assert the exact PTO ISA checkout")

    readme = (root / "docker/README.md").read_text()
    if "docker build -f docker/Dockerfile ." not in readme:
        raise SystemExit("Docker README must document repository-root context")

    workflow = (root / ".github/workflows/isa_contract.yml").read_text()
    if "linxisa-v0.58.3" not in workflow:
        raise SystemExit("raw linxisa-v0.58.3 tag is not an ISA checker trigger")
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
        if 'GITHUB_REF_NAME}" = "linxisa-v0.58.3"' in text:
            raise SystemExit(
                f"{name} conflates ISA identity with PTOAS product version"
            )

    for name in (
        "docker/test_ptoas_cli.sh",
        "docker/collect_ptoas_dist.sh",
        "docker/collect_ptoas_dist_mac.sh",
    ):
        validate_packaging_identity_mode(root / name)

    print("release delivery contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
