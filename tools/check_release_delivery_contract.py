#!/usr/bin/env python3
"""Validate reproducible Docker source and ISA-tag workflow behavior."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptoas-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.ptoas_root.resolve()

    dockerfile = (root / "docker/Dockerfile").read_text()
    if "COPY . ${PTO_SOURCE_DIR}" not in dockerfile:
        raise SystemExit("Docker build must COPY the reviewed PTOAS checkout")
    if "git clone https://github.com/zhangstevenunity/PTOAS.git" in dockerfile:
        raise SystemExit("Docker build must not clone an unpinned PTOAS fork")

    workflow = (root / ".github/workflows/isa_contract.yml").read_text()
    if "linxisa-v0.58.1" not in workflow:
        raise SystemExit("raw linxisa-v0.58.1 tag is not an ISA checker trigger")
    exact_command = "python3 tools/check_v058_pto_manifest.py --ptoas-root ."
    if exact_command not in workflow:
        raise SystemExit("ISA tag workflow does not run the exact contract checker")

    for name in ("build_wheel.yml", "build_wheel_mac.yml"):
        text = (root / ".github/workflows" / name).read_text()
        if 'GITHUB_REF_NAME}" = "linxisa-v0.58.1"' in text:
            raise SystemExit(f"{name} conflates ISA identity with PTOAS product version")

    print("release delivery contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
