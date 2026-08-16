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
import tempfile
import unittest
from pathlib import Path

from check_release_delivery_contract import (
    local_copy_sources,
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

    def test_packaged_cli_checks_exact_pto_isa_identity(self) -> None:
        expected = 'EXPECTED_VERSION_OUTPUT="ptoas ${PTOAS_VERSION} (PTO ISA 0.58.1)"'
        fallback = "grep -Eq '^ptoas [0-9]+\\.[0-9]+ \\(PTO ISA 0\\.58\\.1\\)$'"
        for relative in (
            "docker/test_ptoas_cli.sh",
            "docker/collect_ptoas_dist.sh",
            "docker/collect_ptoas_dist_mac.sh",
        ):
            with self.subTest(script=relative):
                script = (ROOT / relative).read_text()
                self.assertIn(expected, script)
                self.assertIn(fallback, script)


if __name__ == "__main__":
    unittest.main()
