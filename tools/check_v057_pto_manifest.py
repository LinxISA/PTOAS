#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check PTOAS PTO-op contracts against the LinxISA v0.57.1 PTO manifest.

PTOAS is an MLIR PTO dialect-to-EmitC compiler, not a Linx scalar assembler.
This check validates all 120 public operations' exact Linx operand roles/arity,
the matching PTOAS ODS argument surface, and the explicitly enumerated
dialect-only boundary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_LOCK = {
    "release": "0.57.1",
    "encoding_abi": "pto-isa-0.57.1-mode-function-v1",
    "encoding_projection_sha256": "34f6602cf29ea6363d41d896111dad4de0f70ec36517138aa89e292857909da4",
    "content_sha256": "45575361fa5b180c454500360a3e4e75d34fba663e60f3741ef38dc9a514bc13",
    "release_manifest_sha256": "6b72e6242bb971ddc14335884f560e2e97b3c6f8acd7ab0ee2e8b189ca8e243d",
    "source_commit": "b30ed3df4f1a7fd0c2d19b02a90b049cb452fd87",
    "tile_operation_count": 120,
}

DELETED_ACTIVE_NAMES = {
    "TADDC",
    "TADDSC",
    "TFMA",
    "TFMOD",
    "TFMODS",
    "TLRELU",
    "TRANDOM",
    "TSUBC",
    "TSUBSC",
    "TSORT32",
}

def normalize(name: str) -> str:
    return name.replace(".", "").replace("_", "").upper()


def load_lock(ptoas_root: Path) -> None:
    lock_path = ptoas_root / "tools/pto_isa_v0_57_1_lock.json"
    lock = json.loads(lock_path.read_text())
    errors = []
    for key in ("release", "encoding_abi", "encoding_projection_sha256", "content_sha256"):
        if lock.get(key) != EXPECTED_LOCK[key]:
            errors.append(f"{key}: expected {EXPECTED_LOCK[key]}, got {lock.get(key)}")
    if lock.get("release_manifest", {}).get("sha256") != EXPECTED_LOCK["release_manifest_sha256"]:
        errors.append("release_manifest.sha256 mismatch")
    if lock.get("source", {}).get("commit") != EXPECTED_LOCK["source_commit"]:
        errors.append("source.commit mismatch")
    if lock.get("catalogs", {}).get("tile_operations", {}).get("count") != EXPECTED_LOCK["tile_operation_count"]:
        errors.append("catalogs.tile_operations.count mismatch")
    if errors:
        raise SystemExit("unexpected PTO ISA 0.57.1 lock:\n  " + "\n  ".join(errors))


def load_manifest(linx_root: Path) -> dict[str, dict]:
    manifest_path = linx_root / "isa/v0.57/state/pto_ops.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["profile"] != "v0.57" or manifest["operation_count"] != 120:
        raise SystemExit(f"unexpected v0.57.1 manifest header in {manifest_path}")
    source_lock = manifest.get("source_lock")
    if source_lock != "isa/v0.57/pto-spec.lock.json":
        raise SystemExit(f"unexpected v0.57.1 source_lock in {manifest_path}: {source_lock}")
    operations = manifest["operations"]
    names = [entry["name"] for entry in operations]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate operation names in {manifest_path}")
    return {entry["name"]: entry for entry in operations}


def load_expected_contracts(ptoas_root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    contract_path = ptoas_root / "tools/pto_isa_v0_57_1_operation_contracts.json"
    contracts = json.loads(contract_path.read_text())
    if contracts.get("release") != EXPECTED_LOCK["release"]:
        raise SystemExit(f"unexpected release in {contract_path}")
    operations = contracts.get("operations", {})
    if (contracts.get("public_operation_count") != EXPECTED_LOCK["tile_operation_count"] or
            len(operations) != EXPECTED_LOCK["tile_operation_count"]):
        raise SystemExit(f"unexpected public operation count in {contract_path}")
    public = operations
    dialect_only_entries = contracts.get("dialect_only_operations", {})
    dialect_only = {
        normalize(mnemonic): {
            "ptoas_mnemonic": mnemonic,
            "ptoas_arguments": arguments,
        }
        for mnemonic, arguments in dialect_only_entries.items()
    }
    if len(dialect_only) != len(dialect_only_entries):
        raise SystemExit(f"duplicate dialect-only operation names in {contract_path}")
    public_mnemonics = {
        normalize(entry["ptoas_mnemonic"]) for entry in operations.values()
    }
    overlap = sorted(public_mnemonics & set(dialect_only))
    if overlap:
        raise SystemExit(
            f"public/dialect-only operation overlap in {contract_path}: {overlap}"
        )
    return public, dialect_only


def load_ptoas_ops(ptoas_root: Path) -> dict[str, dict]:
    ods_path = ptoas_root / "include/PTO/IR/PTOOps.td"
    if not ods_path.exists():
        ods_path = ptoas_root / "compiler/ptoas/include/PTO/IR/PTOOps.td"
    text = ods_path.read_text()
    definitions = list(
        re.finditer(r'\bdef\s+\w+\s*:\s*PTO_TOp<"([^"]+)"', text)
    )
    operations = {}
    for index, definition in enumerate(definitions):
        mnemonic = definition.group(1)
        start = definition.start()
        end = definitions[index + 1].start() if index + 1 < len(definitions) else len(text)
        body = text[start:end]
        match = re.search(r"let\s+arguments\s*=\s*\(ins(.*?)\);", body, re.DOTALL)
        arguments = tuple(re.findall(r"\$([A-Za-z0-9_]+)", match.group(1))) if match else ()
        key = normalize(mnemonic)
        if key in operations:
            raise SystemExit(f"duplicate normalized PTOAS mnemonic in {ods_path}: {mnemonic}")
        operations[key] = {"mnemonic": mnemonic, "arguments": arguments}
    return operations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptoas-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--linx-root", type=Path)
    args = parser.parse_args()

    ptoas_root = args.ptoas_root.resolve()
    load_lock(ptoas_root)
    expected_public, expected_dialect_only = load_expected_contracts(ptoas_root)
    ptoas_ops = load_ptoas_ops(ptoas_root)

    deleted_present = sorted(name for name in DELETED_ACTIVE_NAMES if normalize(name) in ptoas_ops)
    if deleted_present:
        print("PTOAS has deleted PTO ISA 0.57.1 names active in the dialect:")
        for name in deleted_present:
            print(f"  - {name}")
        return 1

    expected_all = {
        normalize(contract["ptoas_mnemonic"])
        for contract in expected_public.values()
    } | set(expected_dialect_only)
    missing_ops = sorted(expected_all - set(ptoas_ops))
    undeclared_ops = sorted(set(ptoas_ops) - expected_all)
    if missing_ops or undeclared_ops:
        print("PTOAS public/dialect-only operation boundary mismatch:")
        for name in missing_ops:
            print(f"  - missing declared operation: {name}")
        for name in undeclared_ops:
            print(f"  - undeclared dialect-only operation: {name}")
        return 1

    contract_errors = []
    for name, contract in expected_public.items():
        actual = ptoas_ops[normalize(contract["ptoas_mnemonic"])]["arguments"]
        expected = tuple(contract["ptoas_arguments"])
        if actual != expected:
            contract_errors.append(f"{name}: expected PTOAS arguments {expected}, got {actual}")
    for key, contract in expected_dialect_only.items():
        actual = ptoas_ops[key]["arguments"]
        expected = tuple(contract["ptoas_arguments"])
        if actual != expected:
            contract_errors.append(
                f"dialect-only {contract['ptoas_mnemonic']}: "
                f"expected PTOAS arguments {expected}, got {actual}"
            )
    if contract_errors:
        print("PTOAS has incorrect PTO ISA 0.57.1 operation roles/arity:")
        for error in contract_errors:
            print(f"  - {error}")
        return 1

    if args.linx_root is None:
        print(
            "PTOAS v0.57.1 PTO lock/dialect check OK: all 120 public "
            f"operation argument contracts and {len(expected_dialect_only)} explicit "
            "dialect-only contracts match"
        )
        return 0

    manifest = load_manifest(args.linx_root.resolve())

    expected_names = set(expected_public)
    actual_names = set(manifest)
    if actual_names != expected_names:
        print("LinxISA v0.57.1 public operation boundary mismatch:")
        for name in sorted(expected_names - actual_names):
            print(f"  - missing manifest operation: {name}")
        for name in sorted(actual_names - expected_names):
            print(f"  - unexpected manifest operation: {name}")
        return 1

    role_errors = []
    for name, contract in expected_public.items():
        entry = manifest[name]
        actual = tuple(
            (operand.get("field"), operand.get("role"))
            for operand in entry.get("operands", [])
        )
        expected = tuple(
            (operand[0], operand[1])
            for operand in contract["isa_operands"]
        )
        if actual != expected:
            role_errors.append(
                f"{name}: expected ISA operands {expected}, got {actual}"
            )
    if role_errors:
        print("LinxISA v0.57.1 manifest role/arity mismatch:")
        for error in role_errors:
            print(f"  - {error}")
        return 1

    print(
        "PTOAS v0.57.1 PTO manifest contract check OK: "
        f"all {len(manifest)} public operations match exact Linx roles/arity; "
        f"{len(expected_dialect_only)} dialect-only operations are explicitly bounded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
