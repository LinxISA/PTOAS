#!/usr/bin/env python3
"""Audit PTOAS against the unpublished PTO ISA 0.58.7 development snapshot.

PTO-SPEC is the sole architectural golden.  This checker distinguishes mere
ODS/PTO-BC presence from proven semantic compatibility: missing golden
operations must remain unavailable, and operations retired by the golden may
remain parseable only as explicitly rejected dialect compatibility forms.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKER_PATH = Path(__file__).with_name("check_v058_pto_manifest.py")
BASE_CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_v058_pto_manifest", BASE_CHECKER_PATH
)
BASE_CHECKER = importlib.util.module_from_spec(BASE_CHECKER_SPEC)
assert BASE_CHECKER_SPEC.loader is not None
BASE_CHECKER_SPEC.loader.exec_module(BASE_CHECKER)


EXPECTED_IDENTITY = {
    "release": "0.58.7",
    "encoding_abi": "pto-isa-0.58.7-mode-function-v1",
    "encoding_projection_sha256":
        "a5f78ecd707bde4a014542158ea257c783796b95fa4fbcce37742e41fee6a881",
    "status": "development-not-published",
    "publication": None,
    "command_forms_count": 105,
    "command_forms_sha256":
        "d2817054e96456d058d88c27cef1eb034a2b164b486d5c11fe3c796f08e02f3e",
    "scalar_forms_count": 473,
    "scalar_forms_sha256":
        "840ae680f9193ae7a9422a0e7c0e264cd61fa83429149b7b0fff77d41282627d",
    "tile_operations_count": 119,
    "tile_operations_sha256":
        "533db9b8f31f118af5978de813cfaa3fa738de820ee24655faa2ee71f343a706",
    "reservations_count": 50,
    "reservations_sha256":
        "d34f6b560f19a531a962c6d8abf812a911e83ea57f721ef53102ed6f62c9c41c",
    "system_registers_sha256":
        "2a5d6669b95c28f80651f5917558781de25b8465655399063790e50549e1e2a9",
    "development_manifest_sha256":
        "0c6d0b2a1ec0b49e32e13971e751a5ea19946351a3880bfdfcef0f50351944f7",
    "source_commit": "f0a876e449a95d26841c5ee7affb4c905ebc5f5b",
    "source_tree": "fe9e8e050375fbce3f526d08acca13c37cd15830",
}


def normalize(name: str) -> str:
    return name.replace(".", "").replace("_", "").upper()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_capabilities(ptoas_root: Path) -> dict:
    path = ptoas_root / "tools/pto_isa_v0_58_7_capabilities.json"
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise SystemExit(f"unexpected capability schema in {path}")
    if data.get("golden_operation_count") != EXPECTED_IDENTITY["tile_operations_count"]:
        raise SystemExit(f"unexpected golden operation count in {path}")
    if data.get("present_base_contract") != "pto_isa_v0_58_3_operation_contracts.json":
        raise SystemExit(f"unexpected present base contract in {path}")
    if data.get("golden_lock") != "pto_isa_v0_58_7_development_lock.json":
        raise SystemExit(f"unexpected golden lock in {path}")
    retired = data.get("retired_public_operations", [])
    missing_golden = data.get("missing_golden_operations", [])
    for label, names in (("retired", retired), ("missing", missing_golden)):
        if len(names) != len(set(names)):
            raise SystemExit(f"duplicate {label} operation in {path}")
    if set(retired) & set(missing_golden):
        raise SystemExit(f"retired/missing operation overlap in {path}")
    if data.get("present_schema_parity_unproven_count") != 94:
        raise SystemExit(f"unexpected present operation count in {path}")
    drift = data.get("catalog_drift_audit", {})
    if drift.get("reference_source_commit") != "e599a3d36ebfad43362ff591ea5e128816c684c7":
        raise SystemExit(f"unexpected catalog drift reference in {path}")
    if drift.get("exact_catalog_unchanged_count") != 7:
        raise SystemExit(f"unexpected exact-unchanged count in {path}")
    if drift.get("catalog_changed_count") != 87:
        raise SystemExit(f"unexpected catalog-changed count in {path}")
    return data


def load_lock(ptoas_root: Path, capabilities: dict) -> dict:
    path = ptoas_root / "tools" / capabilities["golden_lock"]
    lock = json.loads(path.read_text())
    expected = {
        "status": EXPECTED_IDENTITY["status"],
        "publication": EXPECTED_IDENTITY["publication"],
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise SystemExit(f"unexpected {key} in {path}")
    for key in ("release", "encoding_abi", "encoding_projection_sha256"):
        if lock.get("identity", {}).get(key) != EXPECTED_IDENTITY[key]:
            raise SystemExit(f"unexpected identity.{key} in {path}")
    if lock.get("development_manifest", {}).get("sha256") != EXPECTED_IDENTITY[
        "development_manifest_sha256"
    ]:
        raise SystemExit(f"unexpected development manifest hash in {path}")
    if lock.get("source", {}).get("commit") != EXPECTED_IDENTITY["source_commit"]:
        raise SystemExit(f"unexpected source commit in {path}")
    if lock.get("source", {}).get("tree") != EXPECTED_IDENTITY["source_tree"]:
        raise SystemExit(f"unexpected source tree in {path}")
    for name in ("command_forms", "scalar_forms", "tile_operations", "reservations"):
        entry = lock.get("catalogs", {}).get(name, {})
        if entry.get("sha256") != EXPECTED_IDENTITY[f"{name}_sha256"]:
            raise SystemExit(f"unexpected {name} hash in {path}")
        if entry.get("count") != EXPECTED_IDENTITY[f"{name}_count"]:
            raise SystemExit(f"unexpected {name} count in {path}")
    if lock.get("catalogs", {}).get("system_registers", {}).get(
        "sha256"
    ) != EXPECTED_IDENTITY["system_registers_sha256"]:
        raise SystemExit(f"unexpected system_registers hash in {path}")
    return lock


def load_base_contracts(ptoas_root: Path, capabilities: dict) -> dict:
    path = ptoas_root / "tools" / capabilities["present_base_contract"]
    data = json.loads(path.read_text())
    operations = data.get("operations", {})
    retired = set(capabilities["retired_public_operations"])
    missing = sorted(retired - set(operations))
    if missing:
        raise SystemExit(f"retired operations absent from base contract: {missing}")
    present = {name: value for name, value in operations.items() if name not in retired}
    if len(present) != capabilities["present_schema_parity_unproven_count"]:
        raise SystemExit(f"expected 94 present operations, got {len(present)}")
    return data


def validate_ptoas_boundary(ptoas_root: Path, capabilities: dict, base: dict) -> None:
    operations = BASE_CHECKER.load_ptoas_ops(ptoas_root)
    retired = set(capabilities["retired_public_operations"])
    missing_golden = set(capabilities["missing_golden_operations"])
    present_contracts = {
        name: contract
        for name, contract in base["operations"].items()
        if name not in retired
    }
    base_dialect_only = base.get("dialect_only_operations", {})
    retired_contracts = {name: base["operations"][name] for name in retired}

    expected_present = {
        normalize(contract["ptoas_mnemonic"]): contract
        for contract in present_contracts.values()
    }
    expected_dialect_only = {
        normalize(name): {"ptoas_mnemonic": name, "ptoas_arguments": arguments}
        for name, arguments in base_dialect_only.items()
    }
    expected_dialect_only.update(
        {
            normalize(contract["ptoas_mnemonic"]): contract
            for contract in retired_contracts.values()
        }
    )

    missing_present = sorted(
        name for name in missing_golden if normalize(name) in operations
    )
    if missing_present:
        raise SystemExit(
            "missing PTO ISA 0.58.7 operations became available without "
            f"reviewed semantics: {missing_present}"
        )

    expected_all = set(expected_present) | set(expected_dialect_only)
    missing = sorted(expected_all - set(operations))
    undeclared = sorted(set(operations) - expected_all)
    if missing or undeclared:
        raise SystemExit(
            "PTOAS capability boundary mismatch: "
            f"missing={missing}, undeclared={undeclared}"
        )

    argument_errors = []
    for key, contract in {**expected_present, **expected_dialect_only}.items():
        actual = operations[key]["arguments"]
        expected = tuple(contract["ptoas_arguments"])
        if actual != expected:
            argument_errors.append(
                f"{contract['ptoas_mnemonic']}: expected {expected}, got {actual}"
            )
        if "ptoas_optional_arguments" in contract:
            actual_optional = operations[key]["optional_arguments"]
            expected_optional = tuple(contract["ptoas_optional_arguments"])
            if actual_optional != expected_optional:
                argument_errors.append(
                    f"{contract['ptoas_mnemonic']} optional: expected "
                    f"{expected_optional}, got {actual_optional}"
                )
    if argument_errors:
        raise SystemExit("PTOAS argument mismatch:\n  " + "\n  ".join(argument_errors))

    boundary_source = (ptoas_root / "lib/PTO/IR/PTO.cpp").read_text()
    rejected = set(base.get("linx_rejected_mnemonics", [])) | {
        contract["ptoas_mnemonic"] for contract in retired_contracts.values()
    }
    missing_rejections = sorted(
        mnemonic
        for mnemonic in rejected
        if f'"pto.{mnemonic}"' not in boundary_source
    )
    if missing_rejections:
        raise SystemExit(
            "Linx target does not explicitly reject dialect-only operations: "
            f"{missing_rejections}"
        )

    cli = (ptoas_root / "tools/ptoas/ptoas.cpp").read_text()
    lowering = (ptoas_root / "lib/PTO/Transforms/PTOToEmitC.cpp").read_text()
    boundary_call = "pto::verifyLinxOperationBoundary"
    if boundary_call not in cli:
        raise SystemExit("PTOAS CLI bypasses the shared Linx operation boundary")
    if boundary_call not in lowering:
        raise SystemExit("PTO-to-EmitC pass bypasses the shared Linx operation boundary")

    bytecode = (ptoas_root / "tools/ptobc/generated/ptobc_opcodes_v0.h").read_text()
    bytecode_names = {
        normalize(name)
        for name in re.findall(r'"pto\.([A-Za-z0-9_.]+)"', bytecode)
    }
    missing_bytecode = sorted(
        name for name in missing_golden if normalize(name) in bytecode_names
    )
    if missing_bytecode:
        raise SystemExit(
            "missing PTO ISA 0.58.7 operations have bytecode assignments: "
            f"{missing_bytecode}"
        )


def validate_golden_source(
    source_root: Path, lock: dict, capabilities: dict, base: dict
) -> None:
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != lock["source"]["commit"] or tree != lock["source"]["tree"]:
        raise SystemExit(
            "PTO-SPEC source identity mismatch: "
            f"expected {lock['source']['commit']}/{lock['source']['tree']}, "
            f"got {head}/{tree}"
        )

    manifest_path = source_root / lock["development_manifest"]["path"]
    if sha256(manifest_path) != lock["development_manifest"]["sha256"]:
        raise SystemExit("PTO-SPEC development manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get("identity", {})
    for key in ("release", "encoding_abi", "encoding_projection_sha256"):
        if identity.get(key) != EXPECTED_IDENTITY[key]:
            raise SystemExit(f"development identity {key} mismatch")
    if manifest.get("status") != EXPECTED_IDENTITY["status"]:
        raise SystemExit("PTO-SPEC development status mismatch")
    if manifest.get("publication") is not None:
        raise SystemExit("development snapshot must not claim a publication")

    catalog_keys = {
        "command_forms": "command_forms",
        "scalar_forms": "scalar_forms",
        "tile_operations": "tile_operations",
        "reservations": "reservations",
    }
    for manifest_key, expected_prefix in catalog_keys.items():
        entry = manifest["catalogs"][manifest_key]
        locked_entry = lock["catalogs"][manifest_key]
        expected_hash = EXPECTED_IDENTITY[f"{expected_prefix}_sha256"]
        expected_count = EXPECTED_IDENTITY[f"{expected_prefix}_count"]
        path = source_root / entry["path"]
        if entry != {"path": locked_entry["path"], "sha256": expected_hash}:
            raise SystemExit(f"{manifest_key} lock mismatch")
        if sha256(path) != expected_hash:
            raise SystemExit(f"{manifest_key} hash mismatch")
        if manifest["catalog_counts"][manifest_key] != expected_count:
            raise SystemExit(f"{manifest_key} count mismatch")

    system_registers = manifest["catalogs"]["system_registers"]
    locked_system_registers = lock["catalogs"]["system_registers"]
    if system_registers != locked_system_registers:
        raise SystemExit("system_registers lock mismatch")
    if sha256(source_root / system_registers["path"]) != system_registers["sha256"]:
        raise SystemExit("system_registers hash mismatch")

    tile_path = source_root / manifest["catalogs"]["tile_operations"]["path"]
    tile_entries = json.loads(tile_path.read_text())["operations"]
    tile_by_name = {entry["name"]: entry for entry in tile_entries}
    if len(tile_by_name) != len(tile_entries):
        raise SystemExit("duplicate golden Tile operation name")

    retired = set(capabilities["retired_public_operations"])
    missing_golden = set(capabilities["missing_golden_operations"])
    present = set(base["operations"]) - retired
    if set(tile_by_name) != present | missing_golden:
        raise SystemExit(
            "golden/capability operation boundary mismatch: "
            f"missing={sorted((present | missing_golden) - set(tile_by_name))}, "
            f"unexpected={sorted(set(tile_by_name) - (present | missing_golden))}"
        )

    schema_errors = []
    for name in sorted(present):
        golden = tile_by_name[name]
        contract = base["operations"][name]
        if golden.get("engine") != contract.get("engine"):
            schema_errors.append(
                f"{name}: engine {golden.get('engine')} != {contract.get('engine')}"
            )
        golden_fields = tuple(
            operand["field"] for operand in golden.get("operands", [])
        )
        contract_fields = tuple(operand[0] for operand in contract["isa_operands"])
        if golden_fields != contract_fields:
            schema_errors.append(
                f"{name}: golden fields {golden_fields} != implemented {contract_fields}"
            )
    if schema_errors:
        raise SystemExit(
            "present PTOAS structural operand schema differs from PTO-SPEC golden:\n  "
            + "\n  ".join(schema_errors)
        )

    drift = capabilities["catalog_drift_audit"]
    reference_commit = drift["reference_source_commit"]
    old_tile_text = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "show",
            f"{reference_commit}:spec/catalog/tile-operations.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    old_tile_entries = json.loads(old_tile_text)["operations"]
    old_tile_by_name = {entry["name"]: entry for entry in old_tile_entries}
    exact_unchanged = sorted(
        name for name in present if tile_by_name[name] == old_tile_by_name[name]
    )
    if exact_unchanged != drift["exact_catalog_unchanged_operations"]:
        raise SystemExit(
            "exact-unchanged Tile audit drift: "
            f"expected {drift['exact_catalog_unchanged_operations']}, "
            f"got {exact_unchanged}"
        )

    changed_fields = collections.Counter()
    changed_names = []
    for name in sorted(present):
        old = old_tile_by_name[name]
        new = tile_by_name[name]
        fields = [
            field
            for field in sorted(set(old) | set(new))
            if old.get(field) != new.get(field)
        ]
        if fields:
            changed_names.append(name)
            changed_fields.update(fields)
    if len(changed_names) != drift["catalog_changed_count"]:
        raise SystemExit(
            f"expected {drift['catalog_changed_count']} changed Tile operations, "
            f"got {len(changed_names)}"
        )
    if dict(changed_fields) != drift["changed_field_counts"]:
        raise SystemExit(
            "Tile catalog changed-field audit mismatch: "
            f"expected {drift['changed_field_counts']}, got {dict(changed_fields)}"
        )

    command_path = source_root / manifest["catalogs"]["command_forms"]["path"]
    command_entries = json.loads(command_path.read_text())["forms"]
    old_command_text = subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "show",
            f"{reference_commit}:spec/catalog/command-forms.json",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    old_command_entries = json.loads(old_command_text)["forms"]
    new_ior = [entry for entry in command_entries if entry["mnemonic"] == "B.IOR"]
    old_ior = [entry for entry in old_command_entries if entry["mnemonic"] == "B.IOR"]
    if len(new_ior) != 1 or len(old_ior) != 1:
        raise SystemExit("expected exactly one old and current B.IOR form")
    old_fields = {field["name"] for field in old_ior[0]["fields"]}
    new_fields = {field["name"] for field in new_ior[0]["fields"]}
    if "ExecMaskPresent" in old_fields or "ExecMaskPresent" not in new_fields:
        raise SystemExit("B.IOR ExecMaskPresent drift audit mismatch")
    if old_ior[0]["encoding"] != [
        {"index": 0, "mask": "0x0600707f", "match": "0x00000013", "width_bits": 32}
    ]:
        raise SystemExit("unexpected reference B.IOR encoding")
    if new_ior[0]["encoding"] != [
        {"index": 0, "mask": "0x0200707f", "match": "0x00000013", "width_bits": 32}
    ]:
        raise SystemExit("unexpected current B.IOR encoding")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptoas-root", type=Path, default=ROOT)
    parser.add_argument("--pto-spec-root", type=Path)
    args = parser.parse_args()

    ptoas_root = args.ptoas_root.resolve()
    capabilities = load_capabilities(ptoas_root)
    lock = load_lock(ptoas_root, capabilities)
    base = load_base_contracts(ptoas_root, capabilities)
    validate_ptoas_boundary(ptoas_root, capabilities, base)
    if args.pto_spec_root is not None:
        validate_golden_source(args.pto_spec_root.resolve(), lock, capabilities, base)

    print(
        "PTOAS/PTO-SPEC 0.58.7 development capability check OK: "
        "94 golden Tile operation names are present with semantic/schema parity "
        "unproven, 25 are missing and explicitly unavailable, "
        "15 retired public operations rejected by the Linx target; "
        "PTOAS has no scalar machine-code encoder"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
