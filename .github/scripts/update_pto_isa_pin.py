#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import pathlib
import re
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the pinned pto-isa commit used by CI and Docker."
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="Explicit reviewed implementation-header commit SHA to pin.",
    )
    parser.add_argument(
        "--ci-workflow",
        default=".github/workflows/ci.yml",
        help="Path to the CI workflow file.",
    )
    parser.add_argument(
        "--dockerfile",
        default="docker/Dockerfile",
        help="Path to the Dockerfile that vendors pto-isa.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that all pinned locations already match the target commit.",
    )
    return parser.parse_args()


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_exactly_once(
    text: str, pattern: str, replacement: str, path: pathlib.Path
) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one match for pattern {pattern!r} in {path}, got {count}"
        )
    return new_text


def update_ci_workflow(path: pathlib.Path, commit: str) -> bool:
    original = read_text(path)
    updated = original
    updated = replace_exactly_once(
        updated,
        r"(pto_isa_commit:\n(?:\s+.*\n){0,8}?\s+default:\s*)([0-9a-f]{40})",
        rf"\g<1>{commit}",
        path,
    )
    updated = replace_exactly_once(
        updated,
        r"(PTO_ISA_COMMIT:\s*\$\{\{\s*github\.event\.inputs\.pto_isa_commit\s*\|\|\s*')([^']*)('\s*\}\})",
        rf"\g<1>{commit}\g<3>",
        path,
    )
    if updated != original:
        write_text(path, updated)
        return True
    return False


def update_dockerfile(path: pathlib.Path, commit: str) -> bool:
    original = read_text(path)
    updated = original
    updated = replace_exactly_once(
        updated,
        r"^(ARG PTO_ISA_COMMIT=)([0-9a-f]{40})$",
        rf"\g<1>{commit}",
        path,
    )
    updated = replace_exactly_once(
        updated,
        r"^(# pinned: https://gitcode\.com/cann/pto-isa/commit/)([0-9a-f]{40})$",
        rf"\g<1>{commit}",
        path,
    )
    if updated != original:
        write_text(path, updated)
        return True
    return False


def extract_ci_commit(path: pathlib.Path) -> tuple[str, str]:
    text = read_text(path)
    default_match = re.search(
        r"pto_isa_commit:\n(?:\s+.*\n){0,8}?\s+default:\s*([0-9a-f]{40})",
        text,
        flags=re.MULTILINE,
    )
    env_match = re.search(
        r"PTO_ISA_COMMIT:\s*\$\{\{\s*github\.event\.inputs\.pto_isa_commit\s*\|\|\s*'([0-9a-f]{40})'\s*\}\}",
        text,
    )
    if not default_match or not env_match:
        raise RuntimeError(f"failed to read pinned pto-isa commit from {path}")
    return default_match.group(1), env_match.group(1)


def extract_docker_commit(path: pathlib.Path) -> tuple[str, str]:
    text = read_text(path)
    arg_match = re.search(r"^ARG PTO_ISA_COMMIT=([0-9a-f]{40})$", text, flags=re.MULTILINE)
    comment_match = re.search(
        r"^# pinned: https://gitcode\.com/cann/pto-isa/commit/([0-9a-f]{40})$",
        text,
        flags=re.MULTILINE,
    )
    if not arg_match or not comment_match:
        raise RuntimeError(f"failed to read pinned pto-isa commit from {path}")
    return arg_match.group(1), comment_match.group(1)


def verify(ci_path: pathlib.Path, docker_path: pathlib.Path, commit: str) -> None:
    ci_default, ci_env = extract_ci_commit(ci_path)
    docker_arg, docker_comment = extract_docker_commit(docker_path)
    values = {
        f"{ci_path}:workflow_dispatch_default": ci_default,
        f"{ci_path}:runtime_default": ci_env,
        f"{docker_path}:arg": docker_arg,
        f"{docker_path}:comment": docker_comment,
    }
    mismatches = {name: value for name, value in values.items() if value != commit}
    if mismatches:
        detail = ", ".join(f"{name}={value}" for name, value in mismatches.items())
        raise RuntimeError(f"pto-isa pin mismatch, expected {commit}: {detail}")


def main() -> int:
    args = parse_args()
    commit = args.commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("--commit must be a full 40-character lowercase SHA")
    ci_path = pathlib.Path(args.ci_workflow)
    docker_path = pathlib.Path(args.dockerfile)

    if args.check:
        verify(ci_path, docker_path, commit)
        print(commit)
        return 0

    update_ci_workflow(ci_path, commit)
    update_dockerfile(docker_path, commit)
    verify(ci_path, docker_path, commit)
    print(commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
