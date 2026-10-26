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

"""Check offline dependency discovery and rejection of incompatible checkouts."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "cmake/third_party/opbase.cmake"
CMAKE = shutil.which("cmake")
GIT = shutil.which("git")
PIN = "0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc"


@unittest.skipUnless(CMAKE and GIT, "cmake and git required")
class OpbaseDependencyTest(unittest.TestCase):
    def configure(self, directory, source, use_env=False):
        script = directory / "check.cmake"
        script.write_text(f'include("{MODULE}")\n')
        environment = dict(os.environ)
        if use_env:
            environment["OPBASE_SOURCE_PATH"] = str(source)
        arguments = [] if use_env else [f"-DOPBASE_SOURCE_PATH={source}"]
        return subprocess.run([CMAKE, *arguments, "-P", str(script)],
                              env=environment, capture_output=True, text=True)

    def test_missing_checkout_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result = self.configure(directory, directory / "missing")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pinned opbase Git checkout", result.stderr)
            self.assertIn("README.md", result.stderr)

    def test_wrong_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "opbase"
            subprocess.run([GIT, "init", "-q", str(source)], check=True)
            subprocess.run([GIT, "-C", str(source), "-c", "user.name=Test", "-c",
                            "user.email=test@example.com", "commit", "-q", "--allow-empty", "-m", "fixture"],
                           check=True)
            result = self.configure(directory, source)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(PIN, result.stderr)

    @unittest.skipUnless(os.environ.get("OPBASE_SOURCE_PATH"), "set OPBASE_SOURCE_PATH for pinned checkout checks")
    def test_pinned_checkout_and_local_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "opbase"
            subprocess.run([GIT, "clone", "-q", "--shared", os.environ["OPBASE_SOURCE_PATH"], str(source)],
                           check=True, capture_output=True)
            result = self.configure(directory, source, use_env=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            (source / "unexpected.cpp").write_text("// fixture\n")
            result = self.configure(directory, source)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checkout must be clean", result.stderr)
            (source / "unexpected.cpp").unlink()
            with (source / "LICENSE").open("a") as file:
                file.write("\nfixture\n")
            result = self.configure(directory, source)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checkout must be clean", result.stderr)


if __name__ == "__main__":
    unittest.main()
