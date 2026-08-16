#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check PTOAS PTO-op contracts against the LinxISA v0.58.1 PTO manifest.

PTOAS is an MLIR PTO dialect-to-EmitC compiler, not a Linx scalar assembler.
This check validates all 109 public operations' exact Linx operand roles/arity,
the matching PTOAS ODS argument surface, exact operation engines and roles,
and the bounded dialect-only surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXPECTED_LOCK = {
    "release": "0.58.1",
    "encoding_abi": "pto-isa-0.58.1-mode-function-v1",
    "encoding_projection_sha256": "89b872d6eaf0252200bc9349d49b9346e2a69d894cdcc2dcd0fd71911c1e0b8c",
    "content_sha256": "693e8c0734b48598ac35ffe7fe6f2a01037788fba30ebe026895808d23139f2c",
    "release_manifest_sha256": "acea87af67301173e6d1c6e04014a8dc6e2f658cd2992ed658ab2589cddc7841",
    "source_commit": "c381465b2b8e457e162a4246ee58bb9a2c5b49fd",
    "source_tree": "463a19db3d6ba70022f18bdbca0d4b2c6ed586e4",
    "hardware_profile_path": "spec/hardware-conformance-profile.json",
    "hardware_profile_id": "pto-hardware-numeric-0.58.1-ieee-v1",
    "hardware_profile_sha256": "170deadacb174c933c287231fb67da1046d7989f84b6852bf353d68a495d1755",
    "numeric_vectors_path": "spec/evidence/pto-isa-0581-hardware-numeric-vectors.json",
    "numeric_vectors_sha256": "59c96cc2f45f8e8f3eebb8230338b21ec3a77a99e8fb5e1c7c7b391819a6aa81",
    "command_forms_path": "spec/catalog/command-forms.json",
    "command_forms_sha256": "300a3a57a8728e6c4770da6fff0202b372ec2830edb8dc978dc141d1c26424d0",
    "command_forms_count": 74,
    "scalar_forms_path": "spec/catalog/scalar-forms.json",
    "scalar_forms_sha256": "9f3841d568ffa73fcb43bf4fd365d3c4dba42d27acffa7e273e0f403c0f0c602",
    "scalar_forms_count": 474,
    "tile_operations_path": "spec/catalog/tile-operations.json",
    "tile_operations_sha256": "f163dea8be281fd67173713d373b60f95a9c3c4e558adcdf8034cc213507a1a3",
    "tile_operation_count": 109,
    "extension_encoding_reservations_path": "spec/catalog/extension-encoding-reservations.json",
    "extension_encoding_reservations_sha256": "bdb82b839b98984779d9a1394f6b308f141052ef0b520e5bedb8e87dadd883d4",
    "extension_encoding_reservations_count": 32,
    "release_manifest_path": "spec/release-manifest.json",
    "source_repository": "https://github.com/PTO-ISA/pto-spec.git",
}

DELETED_ACTIVE_NAMES = {
    "ACCCVT",
    "TADDC",
    "TADDSC",
    "TALLOC",
    "TAXPY",
    "TDEINTERLEAVE",
    "TFMOD",
    "TFMODS",
    "TFREE",
    "TGATHERB",
    "TINTERLEAVE",
    "TLRELU",
    "TPARTARGMAX",
    "TPARTARGMIN",
    "TPOP",
    "TPRELU",
    "TPUSH",
    "TRESHAPE",
    "TRANDOM",
    "TSORT32",
    "TSUBC",
    "TSUBSC",
}

def normalize(name: str) -> str:
    return name.replace(".", "").replace("_", "").upper()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lock(ptoas_root: Path) -> dict:
    lock_path = ptoas_root / "tools/pto_isa_v0_58_1_lock.json"
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
        ("tree", "source_tree"),
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
            "scalar_forms",
            ("scalar_forms_path", "scalar_forms_sha256", "scalar_forms_count"),
        ),
        (
            "tile_operations",
            ("tile_operations_path", "tile_operations_sha256", "tile_operation_count"),
        ),
        (
            "extension_encoding_reservations",
            (
                "extension_encoding_reservations_path",
                "extension_encoding_reservations_sha256",
                "extension_encoding_reservations_count",
            ),
        ),
    ):
        entry = catalogs.get(catalog, {})
        for key, expected_key in zip(("path", "sha256", "count"), expected_keys):
            if entry.get(key) != EXPECTED_LOCK[expected_key]:
                errors.append(f"catalogs.{catalog}.{key} mismatch")
    if errors:
        raise SystemExit("unexpected PTO ISA 0.58.1 lock:\n  " + "\n  ".join(errors))
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
    tree = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tree != lock["source"]["tree"]:
        raise SystemExit(
            f"PTO ISA source tree mismatch: expected {lock['source']['tree']}, got {tree}"
        )

    pinned_files = (
        *lock["catalogs"].values(),
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
    manifest_path = linx_root / "isa/v0.58/state/pto_ops.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["profile"] != "v0.58" or manifest["operation_count"] != 109:
        raise SystemExit(f"unexpected v0.58.1 manifest header in {manifest_path}")
    source_lock = manifest.get("source_lock")
    if source_lock != "isa/v0.58/pto-spec.lock.json":
        raise SystemExit(f"unexpected v0.58.1 source_lock in {manifest_path}: {source_lock}")
    operations = manifest["operations"]
    names = [entry["name"] for entry in operations]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate operation names in {manifest_path}")
    return {entry["name"]: entry for entry in operations}


def validate_linx_identity(linx_root: Path, lock: dict) -> None:
    root_lock_path = linx_root / "isa/v0.58/pto-spec.lock.json"
    root_lock = json.loads(root_lock_path.read_text())
    if root_lock != lock:
        local_text = json.dumps(lock, sort_keys=True, indent=2)
        root_text = json.dumps(root_lock, sort_keys=True, indent=2)
        raise SystemExit(
            "PTOAS/root PTO ISA lock mismatch:\n"
            f"--- PTOAS lock\n{local_text}\n--- LinxISA lock\n{root_text}"
        )

    release_path = linx_root / "isa/v0.58/release_manifest.json"
    release = json.loads(release_path.read_text())
    if release.get("profile") != "v0.58" or release.get("version") != lock["release"]:
        raise SystemExit(
            f"LinxISA release manifest mismatch: expected v0.58/{lock['release']}, "
            f"got {release.get('profile')}/{release.get('version')}"
        )
    cardinality = release.get("cardinality", {})
    expected_counts = {
        "command_forms": lock["catalogs"]["command_forms"]["count"],
        "scalar_forms": lock["catalogs"]["scalar_forms"]["count"],
        "tile_operations": lock["catalogs"]["tile_operations"]["count"],
        "extension_encoding_reservations": lock["catalogs"]["extension_encoding_reservations"]["count"],
    }
    for name, expected in expected_counts.items():
        if cardinality.get(name) != expected:
            raise SystemExit(
                f"LinxISA release manifest {name} count mismatch: "
                f"expected {expected}, got {cardinality.get(name)}"
            )


def load_expected_contracts(ptoas_root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    contract_path = ptoas_root / "tools/pto_isa_v0_58_1_operation_contracts.json"
    contract_file = json.loads(contract_path.read_text())
    for field in (
        "release",
        "encoding_abi",
        "encoding_projection_sha256",
        "source_commit",
        "source_tree",
    ):
        if contract_file.get(field) != EXPECTED_LOCK[field]:
            raise SystemExit(f"unexpected {field} in {contract_path}")
    operations = dict(contract_file.get("operations", {}))

    expected_count = EXPECTED_LOCK["tile_operation_count"]
    if contract_file.get("public_operation_count") != expected_count or len(operations) != expected_count:
        raise SystemExit(f"unexpected public operation count in {contract_path}")

    dialect_only_entries = dict(contract_file.get("dialect_only_operations", {}))
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
    return operations, dialect_only


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


def validate_linx_target_surface(ptoas_root: Path) -> None:
    cli_text = (ptoas_root / "tools/ptoas/ptoas.cpp").read_text()
    lowering_text = (ptoas_root / "lib/PTO/Transforms/PTOToEmitC.cpp").read_text()
    errors = []
    for token in (
        'value_desc("a3|a5|linx")',
        'PTOParserTargetArch::Linx',
        'PTOArch::Linx',
        '"jcore/template_asm.hpp"',
    ):
        if token not in cli_text and token not in lowering_text:
            errors.append(f"missing Linx target implementation token: {token}")
    contracts = json.loads(
        (ptoas_root / "tools/pto_isa_v0_58_1_operation_contracts.json").read_text()
    )
    for mnemonic_name in contracts.get("linx_rejected_mnemonics", []):
        mnemonic = f'"pto.{mnemonic_name}"'
        if mnemonic not in cli_text:
            errors.append(f"Linx target boundary does not explicitly reject dialect-only {mnemonic}")
    if 'ValueRange{dst, peerTid, src}' not in lowering_text:
        errors.append("GMOV lowering must match Linx-TileOP-API order (dst, peer_tid, src)")
    if 'ValueRange{dst, src0, src1, src2}' not in lowering_text:
        errors.append("TFMA lowering must match Linx-TileOP-API order (dst, src0, src1, src2)")
    if errors:
        raise SystemExit("invalid PTOAS Linx v0.58.1 target surface:\n  " + "\n  ".join(errors))


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
    validate_linx_target_surface(ptoas_root)
    ptoas_ops = load_ptoas_ops(ptoas_root)

    deleted_present = sorted(
        name
        for name in DELETED_ACTIVE_NAMES
        if normalize(name) in ptoas_ops and normalize(name) not in expected_dialect_only
    )
    boundary_error = bool(deleted_present)
    if deleted_present:
        print("PTOAS has deleted PTO ISA 0.58.1 names active in the dialect:")
        for name in deleted_present:
            print(f"  - {name}")

    expected_all = {
        normalize(contract["ptoas_mnemonic"])
        for contract in expected_public.values()
    } | set(expected_dialect_only)
    missing_ops = sorted(expected_all - set(ptoas_ops))
    undeclared_ops = sorted(set(ptoas_ops) - expected_all)
    if missing_ops or undeclared_ops:
        boundary_error = True
        print("PTOAS public/dialect-only operation boundary mismatch:")
        for name in missing_ops:
            print(f"  - missing declared operation: {name}")
        for name in undeclared_ops:
            print(f"  - undeclared dialect-only operation: {name}")
    if boundary_error:
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
        print("PTOAS has incorrect PTO ISA 0.58.1 operation roles/arity:")
        for error in contract_errors:
            print(f"  - {error}")
        return 1

    if args.linx_root is None:
        print(
            "PTOAS v0.58.1 PTO lock/dialect check OK: all 109 public "
            f"operation argument contracts and {len(expected_dialect_only)} explicit "
            "dialect-only contracts match; hardware numeric profile/vectors are "
            "identity metadata only (execution conformance not evaluated)"
        )
        return 0

    manifest = load_manifest(args.linx_root.resolve())
    validate_linx_identity(args.linx_root.resolve(), lock)

    expected_names = set(expected_public)
    actual_names = set(manifest)
    if actual_names != expected_names:
        print("LinxISA v0.58.1 public operation boundary mismatch:")
        for name in sorted(expected_names - actual_names):
            print(f"  - missing manifest operation: {name}")
        for name in sorted(actual_names - expected_names):
            print(f"  - unexpected manifest operation: {name}")
        return 1

    role_errors = []
    for name, contract in expected_public.items():
        entry = manifest[name]
        if entry.get("engine") != contract.get("engine"):
            role_errors.append(
                f"{name}: expected engine {contract.get('engine')}, got {entry.get('engine')}"
            )
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
        print("LinxISA v0.58.1 manifest role/arity mismatch:")
        for error in role_errors:
            print(f"  - {error}")
        return 1

    print(
        "PTOAS v0.58.1 PTO manifest contract check OK: "
        f"all {len(manifest)} public operations match exact Linx roles/arity; "
        f"{len(expected_dialect_only)} dialect-only operations are explicitly bounded; "
        "hardware numeric profile/vectors are identity metadata only "
        "(execution conformance not evaluated)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
