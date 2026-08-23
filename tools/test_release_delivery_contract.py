#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for the release delivery contract."""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from check_release_delivery_contract import (
    EXPECTED_LLVM_COMMIT,
    EXPECTED_LLVM_TREE,
    EXPECTED_TILEOP_COMMIT,
    EXPECTED_TILEOP_TREE,
    local_copy_sources,
    run_packaging_identity_mode,
    validate_local_copy_sources,
    validate_packaging_identity_mode,
)


ROOT = Path(__file__).resolve().parents[1]


class ReleaseDeliveryContractTest(unittest.TestCase):
    def test_copy_parser_checks_all_local_sources_and_skips_stage_copies(self) -> None:
        dockerfile = "\n".join(
            (
                "COPY first second /sources/",
                "COPY --from=builder /artifact /output",
            )
        )
        self.assertEqual(local_copy_sources(dockerfile), [(1, "first"), (1, "second")])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first").touch()
            with self.assertRaisesRegex(SystemExit, "second"):
                validate_local_copy_sources(root, dockerfile)

    def test_local_docker_copy_sources_exist_from_repository_root(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile").read_text()
        for match in re.finditer(r"^COPY\s+(?!--from=)(\S+)", dockerfile, re.MULTILINE):
            source = match.group(1)
            self.assertTrue(
                (ROOT / source).exists(),
                f"Docker COPY source is not valid from repository-root context: {source}",
            )

    def test_documented_build_uses_repository_root_context(self) -> None:
        readme = (ROOT / "docker/README.md").read_text()
        self.assertIn("docker build -f docker/Dockerfile .", readme)

    def test_docker_llvm_source_build_pins_nanobind_2_9(self) -> None:
        dockerfile = (ROOT / "docker/Dockerfile").read_text()
        self.assertIn("'nanobind>=2.9,<3'", dockerfile)

    def test_all_llvm_source_build_lanes_pin_nanobind_2_9(self) -> None:
        paths = (
            ROOT / "docker/Dockerfile",
            ROOT / ".github/workflows/ci.yml",
            ROOT / ".github/workflows/build_wheel.yml",
            ROOT / ".github/workflows/build_wheel_mac.yml",
        )
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn("nanobind>=2.9,<3", path.read_text())

    def test_all_delivery_lanes_pin_merged_reviewed_llvm(self) -> None:
        paths = (
            ROOT / "README.md",
            ROOT / "docker/Dockerfile",
            ROOT / ".github/workflows/ci.yml",
            ROOT / ".github/workflows/build_wheel.yml",
            ROOT / ".github/workflows/build_wheel_mac.yml",
        )
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn(EXPECTED_LLVM_COMMIT, path.read_text())

    def test_target_integration_pins_merged_llvm_and_tileop_trees(self) -> None:
        text = (ROOT / "CMakeLists.txt").read_text() + (
            ROOT / "README.md"
        ).read_text()
        for identity in (
            EXPECTED_LLVM_COMMIT,
            EXPECTED_LLVM_TREE,
            EXPECTED_TILEOP_COMMIT,
            EXPECTED_TILEOP_TREE,
        ):
            self.assertIn(identity, text)

    def test_hosted_builder_stage_gate_uses_buildkit(self) -> None:
        workflow = (ROOT / ".github/workflows/isa_contract.yml").read_text()
        self.assertIn("docker/setup-buildx-action@", workflow)
        self.assertIn("docker/build-push-action@", workflow)
        self.assertRegex(workflow, r"(?m)^\s+context:\s+\.\s*$")
        self.assertRegex(workflow, r"(?m)^\s+file:\s+docker/Dockerfile\s*$")
        self.assertRegex(workflow, r"(?m)^\s+target:\s+builder\s*$")
        self.assertRegex(workflow, r"(?m)^\s+push:\s+false\s*$")

    def test_packaged_cli_identity_validator_is_fail_closed(self) -> None:
        scripts = (
            ROOT / "docker/test_ptoas_cli.sh",
            ROOT / "docker/collect_ptoas_dist.sh",
            ROOT / "docker/collect_ptoas_dist_mac.sh",
        )

        def run(
            script: Path, output: str, product_version: str = ""
        ) -> subprocess.CompletedProcess[str]:
            return run_packaging_identity_mode(script, output, product_version)

        invalid_outputs = (
            "ptoas 0.41",
            "ptoas 0.40 (PTO ISA 0.58.3)",
            "ptoas 0.41 (PTO ISA 0.58.0)",
            "warning\nptoas 0.41 (PTO ISA 0.58.3)",
            "ptoas 0.41 (PTO ISA 0.58.3)\nptoas 0.40",
        )
        for script in scripts:
            with self.subTest(
                script=script.name, output="valid", product_version="0.41"
            ):
                self.assertEqual(
                    run(script, "ptoas 0.41 (PTO ISA 0.58.3)", "0.41").returncode,
                    0,
                )
            with self.subTest(script=script.name, output="valid", product_version=""):
                self.assertEqual(
                    run(script, "ptoas 0.41 (PTO ISA 0.58.3)").returncode, 0
                )
            for product_version in ("0.41", ""):
                for invalid in invalid_outputs:
                    with self.subTest(
                        script=script.name,
                        output=invalid,
                        product_version=product_version,
                    ):
                        self.assertNotEqual(
                            run(script, invalid, product_version).returncode, 0
                        )

    def test_dead_identity_entrypoints_fail_dynamic_validation(self) -> None:
        dead_scripts = (
            "function dead_check { exit 0; }; exit 1",
            "false && { exit 0; }; exit 1",
            "cat <<'DEAD-TEXT'\nexit 0\nDEAD-TEXT\nexit 1",
            "cat <<\\DEAD\nexit 0\nDEAD\nexit 1",
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, body in enumerate(dead_scripts):
                script = Path(directory) / f"dead-{index}.sh"
                script.write_text(f"#!/usr/bin/env bash\n{body}\n")
                with self.subTest(body=body), self.assertRaises(SystemExit):
                    validate_packaging_identity_mode(script)

    def test_shared_production_identity_call_cannot_be_removed_or_bypassed(
        self,
    ) -> None:
        shared_call = (
            'bash "${SCRIPT_DIR}/check_ptoas_cli_identity.sh" '
            '"${VERSION_OUTPUT}" "${PTOAS_VERSION:-}"'
        )
        helper_text = (ROOT / "docker/check_ptoas_cli_identity.sh").read_text()
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            (temp_root / "check_ptoas_cli_identity.sh").write_text(helper_text)
            for source in (
                ROOT / "docker/test_ptoas_cli.sh",
                ROOT / "docker/collect_ptoas_dist.sh",
                ROOT / "docker/collect_ptoas_dist_mac.sh",
            ):
                original = source.read_text()
                self.assertEqual(original.count(shared_call), 1)
                for label, replacement in (
                    ("removed", ": # identity validation removed"),
                    ("bypassed", f"false && {shared_call}"),
                ):
                    mutated = temp_root / f"{source.stem}-{label}.sh"
                    mutated.write_text(original.replace(shared_call, replacement, 1))
                    with (
                        self.subTest(script=source.name, mutation=label),
                        self.assertRaises(SystemExit),
                    ):
                        validate_packaging_identity_mode(mutated)


if __name__ == "__main__":
    unittest.main()
