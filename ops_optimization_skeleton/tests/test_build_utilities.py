#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 multimodal-bench contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Host regression checks for the kernel metadata used by the Toolkit build."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/util"))


class BuildUtilitiesTest(unittest.TestCase):
    def test_kernel_source_selection(self):
        module = importlib.import_module("insert_kernel_src")
        with tempfile.TemporaryDirectory() as directory:
            ini = Path(directory) / "aic-ascend910b-ops-info.ini"
            ini.write_text("[Logit]\ninput0.name=x\n[Other]\ninput0.name=z\n")
            module.insert_ini_kernel_src(
                "Logit ALL arch/default,Logit ascend910b arch/specific,Other ascend910 arch/skipped",
                directory, "ascend910b")
            self.assertEqual(ini.read_text(), "[Logit]\nkernelSrc.value=arch/specific\n"
                             "input0.name=x\n[Other]\ninput0.name=z\n")


if __name__ == "__main__":
    unittest.main()
