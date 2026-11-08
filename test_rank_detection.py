"""Small linear systems with known rank and shared-mean repair limits."""

import unittest

import torch

try:
    from .rank_detection import _analyze_time, _linearized_repair
except ImportError:
    from rank_detection import _analyze_time, _linearized_repair


class RankDetectionTests(unittest.TestCase):
    def test_full_rank_missing_block_is_locally_repairable(self) -> None:
        jacobian = torch.diag(torch.tensor([2.0, 0.5]))
        mismatch = torch.tensor([4.0, -1.0])
        result, step, _ = _linearized_repair(
            jacobian, mismatch,
            relative_tolerance=1e-3, absolute_tolerance=1e-6,
        )
        self.assertEqual(result["numerical_rank"], 2)
        self.assertAlmostEqual(result["linearized_residual_ratio"], 0.0)
        torch.testing.assert_close(step, torch.tensor([-2.0, 2.0], dtype=torch.float64))

    def test_shared_mean_conflict_is_separate_from_local_rank(self) -> None:
        # Each spot can repair its one missing gene, but they require opposite steps.
        jacobian = torch.zeros(4, 2)
        jacobian[1, 1] = jacobian[3, 1] = 1.0
        mismatch = torch.tensor([0.0, 1.0, 0.0, -1.0])
        result, _, _, _ = _analyze_time(
            jacobian, mismatch,
            missing_mask=torch.tensor([False, True]),
            spot_indices=[3, 7],
            relative_tolerance=1e-3, absolute_tolerance=1e-6,
        )
        self.assertEqual(result["shared_mu"]["numerical_rank"], 1)
        self.assertAlmostEqual(result["shared_mu"]["linearized_residual_ratio"], 1.0)
        self.assertAlmostEqual(result["per_spot_mu"]["linearized_residual_ratio"], 0.0)
        self.assertAlmostEqual(result["shared_minus_per_spot_residual_ratio"], 1.0)

    def test_zero_missing_rank_cannot_repair_mismatch(self) -> None:
        result, _, _, _ = _analyze_time(
            torch.zeros(4, 2), torch.tensor([0.0, 2.0, 0.0, 1.0]),
            missing_mask=torch.tensor([False, True]),
            spot_indices=[3, 7],
            relative_tolerance=1e-3, absolute_tolerance=1e-6,
        )
        self.assertEqual(result["shared_mu"]["numerical_rank"], 0)
        self.assertAlmostEqual(result["shared_mu"]["linearized_residual_ratio"], 1.0)
        self.assertAlmostEqual(result["per_spot_mu"]["linearized_residual_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
