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
    has_top_level_identity_invocation,
    local_copy_sources,
    top_level_shell_lines,
    validate_local_copy_sources,
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

    def test_hosted_builder_stage_gate_uses_buildkit(self) -> None:
        workflow = (ROOT / ".github/workflows/isa_contract.yml").read_text()
        self.assertIn("docker/setup-buildx-action@", workflow)
        self.assertIn("docker/build-push-action@", workflow)
        self.assertRegex(workflow, r"(?m)^\s+context:\s+\.\s*$")
        self.assertRegex(workflow, r"(?m)^\s+file:\s+docker/Dockerfile\s*$")
        self.assertRegex(workflow, r"(?m)^\s+target:\s+builder\s*$")
        self.assertRegex(workflow, r"(?m)^\s+push:\s+false\s*$")

    def test_packaged_cli_identity_validator_is_fail_closed(self) -> None:
        helper = ROOT / "docker/check_ptoas_cli_identity.sh"

        def run(
            output: str, product_version: str = ""
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["bash", str(helper), output, product_version],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(run("ptoas 0.41 (PTO ISA 0.58.1)", "0.41").returncode, 0)
        self.assertEqual(run("ptoas 0.41 (PTO ISA 0.58.1)").returncode, 0)
        invalid_outputs = (
            "ptoas 0.41",
            "ptoas 0.40 (PTO ISA 0.58.1)",
            "ptoas 0.41 (PTO ISA 0.58.0)",
            "warning\nptoas 0.41 (PTO ISA 0.58.1)",
            "ptoas 0.41 (PTO ISA 0.58.1)\nptoas 0.40",
        )
        for product_version in ("0.41", ""):
            for invalid in invalid_outputs:
                with self.subTest(output=invalid, product_version=product_version):
                    self.assertNotEqual(run(invalid, product_version).returncode, 0)

    def test_packaging_scripts_execute_identity_validator(self) -> None:
        invocation = (
            'bash "${PTO_SOURCE_DIR}/docker/check_ptoas_cli_identity.sh" '
            '"${VERSION_OUTPUT}" "${PTOAS_VERSION:-}"'
        )
        for relative in (
            "docker/test_ptoas_cli.sh",
            "docker/collect_ptoas_dist.sh",
            "docker/collect_ptoas_dist_mac.sh",
        ):
            with self.subTest(script=relative):
                script = (ROOT / relative).read_text()
                self.assertTrue(has_top_level_identity_invocation(script, invocation))

        dead_branch = "\n".join(
            ('echo "$VERSION_OUTPUT"', "if false; then", invocation, "fi")
        )
        heredoc = "\n".join(
            ('echo "$VERSION_OUTPUT"', "cat <<'DEAD'", invocation, "DEAD")
        )
        dead_function = "\n".join(
            (
                'echo "$VERSION_OUTPUT"',
                "dead_check() {",
                invocation,
                "}",
            )
        )
        for script in (f"# {invocation}", dead_branch, heredoc, dead_function):
            with self.subTest(script=script):
                self.assertFalse(has_top_level_identity_invocation(script, invocation))

    def test_top_level_shell_parser_preserves_live_lines_after_heredocs(self) -> None:
        script = "\n".join(("cat <<'PY'", "if false; then", "PY", "echo live"))
        self.assertEqual(top_level_shell_lines(script), ["cat <<'PY'", "echo live"])


if __name__ == "__main__":
    unittest.main()
