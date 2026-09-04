import math
import unittest

import numpy as np
import torch

from vlact_ext.wrap_aware_loss import (
    TWO_PI,
    flow_matching_sample_estimate,
    masked_wrap_aware_l1,
    masked_wrap_aware_l1_np,
    wrap_aware_residual,
    wrap_to_pi,
)

DEG = math.pi / 180.0


class TestWrapToPi(unittest.TestCase):
    def test_wraps_into_half_open_interval_torch_and_numpy(self):
        x = np.array([0.0, math.pi, -math.pi, 3 * math.pi, -3.5 * math.pi, 190 * DEG])
        expected = np.array([0.0, -math.pi, -math.pi, -math.pi, 0.5 * math.pi, -170 * DEG])
        np.testing.assert_allclose(wrap_to_pi(x), expected, atol=1e-12)
        torch.testing.assert_close(wrap_to_pi(torch.tensor(x)), torch.tensor(expected), atol=1e-12, rtol=0)

    def test_custom_period(self):
        # normalised joints: [-pi, pi] -> [-1, 1] means one turn is 2.0
        self.assertAlmostEqual(float(wrap_to_pi(np.array(1.5), period=2.0)), -0.5)
        self.assertAlmostEqual(float(wrap_to_pi(torch.tensor(-1.25), period=2.0)), 0.75)

    def test_per_dimension_period(self):
        x = torch.tensor([[1.5, 1.5]])
        out = wrap_to_pi(x, period=torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(out, torch.tensor([[-0.5, 1.5]]))


class TestMaskedWrapAwareL1(unittest.TestCase):
    def test_boundary_179_vs_minus_179_degrees(self):
        pred = torch.tensor([[[179 * DEG]]])
        target = torch.tensor([[[-179 * DEG]]])
        periodic = torch.tensor([True])
        loss = masked_wrap_aware_l1(pred, target, active_mask=None, periodic_mask=periodic)
        self.assertAlmostEqual(loss.item(), 2 * DEG, places=6)
        self.assertAlmostEqual(loss.item(), 0.0349, places=4)
        # without the periodic flag the same pair costs almost a full turn
        plain = masked_wrap_aware_l1(pred, target, active_mask=None, periodic_mask=None)
        self.assertAlmostEqual(plain.item(), 358 * DEG, places=6)

    def test_non_periodic_dims_degenerate_to_plain_l1(self):
        torch.manual_seed(0)
        pred = torch.randn(2, 3, 4) * 10
        target = torch.randn(2, 3, 4) * 10
        periodic = torch.zeros(4, dtype=torch.bool)
        loss = masked_wrap_aware_l1(pred, target, None, periodic)
        torch.testing.assert_close(loss, (pred - target).abs().mean())
        torch.testing.assert_close(masked_wrap_aware_l1(pred, target), (pred - target).abs().mean())

    def test_mixed_periodic_and_plain_dims(self):
        pred = torch.tensor([[[179 * DEG, 5.0]]])
        target = torch.tensor([[[-179 * DEG, 1.0]]])
        periodic = torch.tensor([True, False])
        loss = masked_wrap_aware_l1(pred, target, None, periodic)
        self.assertAlmostEqual(loss.item(), (2 * DEG + 4.0) / 2, places=6)

    def test_masked_dims_do_not_contribute(self):
        pred = torch.zeros(2, 3, 4)
        target = torch.zeros(2, 3, 4)
        target[..., 2] = 1e6  # garbage in a masked slot
        target[..., 0] = 1.0
        active = torch.tensor([True, True, False, True])
        loss = masked_wrap_aware_l1(pred, target, active_mask=active)
        # 2*3 cells with error 1.0 out of 2*3*3 active cells
        self.assertAlmostEqual(loss.item(), 6.0 / 18.0, places=6)
        self.assertTrue(torch.isfinite(loss))

    def test_all_masked_returns_zero_without_nan_and_keeps_graph(self):
        pred = torch.randn(2, 3, 4, requires_grad=True)
        target = torch.randn(2, 3, 4)
        active = torch.zeros(2, 3, 4, dtype=torch.bool)
        loss = masked_wrap_aware_l1(pred, target, active_mask=active, periodic_mask=torch.ones(4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertTrue(torch.all(pred.grad == 0))

    def test_mask_broadcast_shapes(self):
        pred = torch.zeros(2, 3, 4)
        target = torch.ones(2, 3, 4)
        for mask in (torch.ones(4, dtype=torch.bool), torch.ones(3, 4, dtype=torch.bool), torch.ones(2, 1, 4, dtype=torch.bool)):
            self.assertAlmostEqual(masked_wrap_aware_l1(pred, target, mask).item(), 1.0)
        with self.assertRaises(ValueError):
            masked_wrap_aware_l1(pred, target, torch.ones(5, dtype=torch.bool))
        with self.assertRaises(ValueError):
            masked_wrap_aware_l1(pred, torch.ones(2, 3, 5))

    def test_gradient_flows_through_wrapped_residual(self):
        pred = torch.tensor([[[179 * DEG]]], requires_grad=True)
        target = torch.tensor([[[-179 * DEG]]])
        loss = masked_wrap_aware_l1(pred, target, None, torch.tensor([True]))
        loss.backward()
        # wrapped residual is -2deg: pushing pred towards +180deg (== -180deg) shrinks the loss
        self.assertAlmostEqual(pred.grad.item(), -1.0, places=6)

    def test_period_two_normalised_space(self):
        pred = torch.tensor([[[0.99]]])
        target = torch.tensor([[[-0.99]]])
        loss = masked_wrap_aware_l1(pred, target, None, torch.tensor([True]), period=2.0)
        self.assertAlmostEqual(loss.item(), 0.02, places=6)


class TestNumpyTwin(unittest.TestCase):
    def test_matches_torch_version(self):
        rng = np.random.default_rng(0)
        pred = rng.uniform(-4, 4, size=(3, 5, 6))
        target = rng.uniform(-4, 4, size=(3, 5, 6))
        active = rng.uniform(size=(3, 5, 6)) > 0.3
        periodic = np.array([True, True, False, False, True, False])
        ref = masked_wrap_aware_l1(
            torch.tensor(pred), torch.tensor(target), torch.tensor(active), torch.tensor(periodic)
        ).item()
        self.assertAlmostEqual(masked_wrap_aware_l1_np(pred, target, active, periodic), ref, places=10)

    def test_boundary_and_all_masked(self):
        pred = np.array([[[179 * DEG]]])
        target = np.array([[[-179 * DEG]]])
        self.assertAlmostEqual(masked_wrap_aware_l1_np(pred, target, None, np.array([True])), 2 * DEG, places=10)
        self.assertEqual(masked_wrap_aware_l1_np(pred, target, np.array([False]), np.array([True])), 0.0)

    def test_residual_helper(self):
        delta = wrap_aware_residual(np.array([3.0]), np.array([-3.0]), np.array([True]))
        self.assertAlmostEqual(float(delta[0]), 6.0 - TWO_PI, places=12)
        delta_plain = wrap_aware_residual(np.array([3.0]), np.array([-3.0]), None)
        self.assertAlmostEqual(float(delta_plain[0]), 6.0, places=12)


class TestFlowMatchingSampleEstimate(unittest.TestCase):
    def test_recovers_clean_sample_from_true_velocity(self):
        torch.manual_seed(0)
        a = torch.randn(2, 4, 3)
        eps = torch.randn(2, 4, 3)
        t = torch.rand(2, 1, 1)
        x_t = (1 - t) * eps + t * a
        torch.testing.assert_close(flow_matching_sample_estimate(x_t, a - eps, t), a, atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
