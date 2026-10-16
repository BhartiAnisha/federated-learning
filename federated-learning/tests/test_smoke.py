"""Smoke test: a --quick run on the hospital dataset finishes and writes its outputs."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class QuickRunTest(unittest.TestCase):
    def test_quick_hospital_run(self):
        with tempfile.TemporaryDirectory() as out:
            subprocess.run(
                [sys.executable, "main.py", "--quick", "--datasets", "hospital", "--results-dir", out],
                cwd=ROOT, check=True, capture_output=True, timeout=600,
            )
            self.assertTrue((Path(out) / "raw_hospital.csv").exists())
            self.assertTrue((Path(out) / "best_config_hospital.json").exists())


if __name__ == "__main__":
    unittest.main()
