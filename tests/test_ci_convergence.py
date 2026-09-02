"""Fast CPU tests for the TP/PP Eagle3 convergence comparator."""

import importlib.util
import unittest
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "tools" / "ci" / "compare_eagle3_convergence.py"
_SPEC = importlib.util.spec_from_file_location("compare_eagle3_convergence", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestCompareTrajectories(unittest.TestCase):
    def test_matching_trajectories_pass(self):
        result = _MODULE.compare_trajectories(
            [{"step": 1, "loss": 10.0}, {"step": 2, "loss": 5.0}],
            [{"step": 1, "loss": 10.05}, {"step": 2, "loss": 5.02}],
        )
        self.assertAlmostEqual(result["worst"]["relative_diff"], 0.05 / 10.05, places=6)

    def test_divergent_trajectory_fails(self):
        with self.assertRaisesRegex(ValueError, "diverged"):
            _MODULE.compare_trajectories(
                [{"step": 1, "loss": 10.0}],
                [{"step": 1, "loss": 12.0}],
            )


if __name__ == "__main__":
    unittest.main()
