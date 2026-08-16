#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("check_v058_pto_manifest.py")
SPEC = importlib.util.spec_from_file_location("check_v058_pto_manifest", MODULE_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKER)


class LinxIdentityTest(unittest.TestCase):
    def test_mismatched_release_fails_closed(self):
        local_lock = json.loads(
            Path(__file__).with_name("pto_isa_v0_58_1_lock.json").read_text()
        )
        mismatched = json.loads(json.dumps(local_lock))
        mismatched["release"] = "0.58.0"
        with tempfile.TemporaryDirectory() as directory:
            linx_root = Path(directory)
            lock_path = linx_root / "isa/v0.58/pto-spec.lock.json"
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text(json.dumps(mismatched))
            with self.assertRaisesRegex(SystemExit, "PTOAS/root PTO ISA lock mismatch"):
                CHECKER.validate_linx_identity(linx_root, local_lock)


if __name__ == "__main__":
    unittest.main()
