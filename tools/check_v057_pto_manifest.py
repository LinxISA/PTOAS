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
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED_LOCK = {
    "release": "0.57.1",
    "encoding_abi": "pto-isa-0.57.1-mode-function-v1",
    "encoding_projection_sha256": "34f6602cf29ea6363d41d896111dad4de0f70ec36517138aa89e292857909da4",
    "content_sha256": "5d75f42d191478aef9fa1ef1d73fb18dc48cf83468dc79c9054cb4ae21387354",
    "release_manifest_sha256": "d0aa98754622d7949e55cccd576dd3da74bceb8c060e853cccf1f84c13b4438a",
    "source_commit": "0141ec89ff1e222adfe1df55610f414ca0c8c086",
    "hardware_profile_path": "spec/hardware-conformance-profile.json",
    "hardware_profile_id": "pto-hardware-numeric-0.57.1-ieee-v1",
    "hardware_profile_sha256": "becfbefcc7a31408e5e5293802493f833f68a2fccebb3f9e8008d8b9f4e7658c",
    "numeric_vectors_path": "spec/evidence/pto-isa-0571-hardware-numeric-vectors.json",
    "numeric_vectors_sha256": "b9f908deb9cfec412388e95f4532563cf4e462032b43d15996f0f7b4b93b4ec9",
    "command_forms_path": "spec/catalog/command-forms.json",
    "command_forms_sha256": "c53db18b30fbf53676f1d733e215122f65ad681a778ae0728cc6c4a3674df61e",
    "command_forms_count": 99,
    "tile_operations_path": "spec/catalog/tile-operations.json",
    "tile_operations_sha256": "2a49616fbbd34ee4ff00b971d56de6dd7b8c1698fa7312db0b20b6119965bc26",
    "tile_operation_count": 120,
    "release_manifest_path": "spec/release-manifest.json",
    "source_repository": "https://github.com/PTO-ISA/pto-spec.git",
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lock(ptoas_root: Path) -> dict:
    lock_path = ptoas_root / "tools/pto_isa_v0_57_1_lock.json"
    lock = json.loads(lock_path.read_text())
    errors = []
    for key in ("release", "encoding_abi", "encoding_projection_sha256", "content_sha256"):
        if lock.get(key) != EXPECTED_LOCK[key]:
            errors.append(f"{key}: expected {EXPECTED_LOCK[key]}, got {lock.get(key)}")
    release_manifest = lock.get("release_manifest", {})
    for key, expected_key in (
        ("path", "release_manifest_path"),
        ("sha256", "release_manifest_sha256"),
    ):
        if release_manifest.get(key) != EXPECTED_LOCK[expected_key]:
            errors.append(f"release_manifest.{key} mismatch")
    source = lock.get("source", {})
    for key, expected_key in (
        ("commit", "source_commit"),
        ("repository", "source_repository"),
    ):
        if source.get(key) != EXPECTED_LOCK[expected_key]:
            errors.append(f"source.{key} mismatch")
    hardware_profile = lock.get("hardware_conformance_profile", {})
    for key, expected_key in (
        ("path", "hardware_profile_path"),
        ("profile_id", "hardware_profile_id"),
        ("sha256", "hardware_profile_sha256"),
    ):
        if hardware_profile.get(key) != EXPECTED_LOCK[expected_key]:
            errors.append(f"hardware_conformance_profile.{key} mismatch")
    numeric_vectors = lock.get("numeric_conformance_vectors", {})
    for key, expected_key in (
        ("path", "numeric_vectors_path"),
        ("sha256", "numeric_vectors_sha256"),
    ):
        if numeric_vectors.get(key) != EXPECTED_LOCK[expected_key]:
            errors.append(f"numeric_conformance_vectors.{key} mismatch")
    catalogs = lock.get("catalogs", {})
    for catalog, expected_keys in (
        (
            "command_forms",
            ("command_forms_path", "command_forms_sha256", "command_forms_count"),
        ),
        (
            "tile_operations",
            ("tile_operations_path", "tile_operations_sha256", "tile_operation_count"),
        ),
    ):
        entry = catalogs.get(catalog, {})
        for key, expected_key in zip(("path", "sha256", "count"), expected_keys):
            if entry.get(key) != EXPECTED_LOCK[expected_key]:
                errors.append(f"catalogs.{catalog}.{key} mismatch")
    if errors:
        raise SystemExit("unexpected PTO ISA 0.57.1 lock:\n  " + "\n  ".join(errors))
    return lock


def validate_source_tree(source_root: Path, lock: dict) -> None:
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot resolve PTO ISA source commit in {source_root}: {error}")
    if head != lock["source"]["commit"]:
        raise SystemExit(
            f"PTO ISA source commit mismatch: expected {lock['source']['commit']}, got {head}"
        )

    pinned_files = (
        lock["catalogs"]["command_forms"],
        lock["catalogs"]["tile_operations"],
        lock["release_manifest"],
        lock["hardware_conformance_profile"],
        lock["numeric_conformance_vectors"],
    )
    for entry in pinned_files:
        path = source_root / entry["path"]
        if not path.is_file():
            raise SystemExit(f"missing pinned PTO ISA source file: {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise SystemExit(
                f"PTO ISA source hash mismatch for {entry['path']}: "
                f"expected {entry['sha256']}, got {actual}"
            )

    manifest = json.loads((source_root / lock["release_manifest"]["path"]).read_text())
    expected_manifest_fields = {
        "release": lock["release"],
        "encoding_abi": lock["encoding_abi"],
        "encoding_projection_sha256": lock["encoding_projection_sha256"],
        "content_sha256": lock["content_sha256"],
    }
    for key, expected in expected_manifest_fields.items():
        if manifest.get(key) != expected:
            raise SystemExit(
                f"PTO ISA release manifest {key} mismatch: "
                f"expected {expected}, got {manifest.get(key)}"
            )
    hardware = manifest.get("hardware_conformance_profile", {})
    for key in ("path", "profile_id", "sha256"):
        if hardware.get(key) != lock["hardware_conformance_profile"][key]:
            raise SystemExit(f"PTO ISA release manifest hardware profile {key} mismatch")
    if hardware.get("evidence") != lock["numeric_conformance_vectors"]["path"]:
        raise SystemExit("PTO ISA release manifest hardware numeric evidence path mismatch")
    owned_hashes = {
        entry.get("path"): entry.get("sha256")
        for entry in manifest.get("owned_artifacts", [])
    }
    numeric_vectors = lock["numeric_conformance_vectors"]
    if owned_hashes.get(numeric_vectors["path"]) != numeric_vectors["sha256"]:
        raise SystemExit("PTO ISA release manifest hardware numeric evidence hash mismatch")
    if (
        manifest.get("catalog_counts", {}).get("tile_operations_total")
        != lock["catalogs"]["tile_operations"]["count"]
    ):
        raise SystemExit("PTO ISA release manifest tile operation count mismatch")
    if (
        manifest.get("catalog_counts", {}).get("command_forms")
        != lock["catalogs"]["command_forms"]["count"]
    ):
        raise SystemExit("PTO ISA release manifest command form count mismatch")


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
    parser.add_argument(
        "--pto-spec-root",
        type=Path,
        help="optionally verify every pinned identity against a PTO-ISA/pto-spec checkout",
    )
    args = parser.parse_args()

    ptoas_root = args.ptoas_root.resolve()
    lock = load_lock(ptoas_root)
    if args.pto_spec_root is not None:
        validate_source_tree(args.pto_spec_root.resolve(), lock)
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
            "dialect-only contracts match; hardware numeric profile/vectors are "
            "identity metadata only (execution conformance not evaluated)"
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
        f"{len(expected_dialect_only)} dialect-only operations are explicitly bounded; "
        "hardware numeric profile/vectors are identity metadata only "
        "(execution conformance not evaluated)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
