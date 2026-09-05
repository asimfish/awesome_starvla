"""CPU tests for FutureFeaturePredictionHead, feature_prediction_loss and targets_from_sequence."""

import unittest

import torch

from starvla_lab.heads import FutureFeaturePredictionHead, feature_prediction_loss, masked_mean, targets_from_sequence

B, K, D, D_FEAT = 3, 4, 16, 8


class TestFeaturePredictionLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.target = torch.randn(B, 2, D_FEAT)

    def test_perfect_prediction_is_zero(self):
        out = feature_prediction_loss(self.target.clone(), self.target)
        self.assertLess(out["loss"].item(), 1e-5)
        self.assertLess(out["cosine_loss"].item(), 1e-5)
        self.assertEqual(out["mse_loss"].item(), 0.0)

    def test_terms_and_weights(self):
        pred = torch.randn(B, 2, D_FEAT)
        out = feature_prediction_loss(pred, self.target, cosine_weight=0.5, mse_weight=2.0)
        torch.testing.assert_close(out["loss"], 0.5 * out["cosine_loss"] + 2.0 * out["mse_loss"])
        # 1 - cos is bounded by 2 and the MSE term is a plain mean of squared errors
        self.assertLessEqual(out["cosine_loss"].item(), 2.0)
        torch.testing.assert_close(out["mse_loss"], ((pred - self.target) ** 2).mean())
        anti = feature_prediction_loss(-self.target, self.target, mse_weight=0.0)
        torch.testing.assert_close(anti["loss"], torch.tensor(2.0), atol=1e-5, rtol=0)

    def test_mask_removes_entries(self):
        pred = self.target.clone()
        pred[:, 1] = 1e3  # garbage only where the mask is False
        mask = torch.tensor([[True, False]] * B)
        out = feature_prediction_loss(pred, self.target, mask)
        self.assertLess(out["loss"].item(), 1e-5)
        self.assertGreater(feature_prediction_loss(pred, self.target)["loss"].item(), 1.0)
        # mask is respected per entry, not per row
        mask[0, 1] = True
        self.assertGreater(feature_prediction_loss(pred, self.target, mask)["loss"].item(), 1.0)

    def test_all_false_mask_is_zero_without_nan_and_backprops(self):
        pred = torch.randn(B, 2, D_FEAT, requires_grad=True)
        out = feature_prediction_loss(pred, self.target, torch.zeros(B, 2, dtype=torch.bool))
        self.assertEqual(out["loss"].item(), 0.0)
        self.assertFalse(torch.isnan(out["loss"]))
        out["loss"].backward()
        self.assertTrue(torch.all(pred.grad == 0))
        self.assertEqual(masked_mean(torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.bool)).item(), 0.0)

    def test_zero_target_does_not_nan(self):
        out = feature_prediction_loss(torch.randn(B, 1, D_FEAT), torch.zeros(B, 1, D_FEAT))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_shape_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            feature_prediction_loss(torch.zeros(B, 1, D_FEAT), torch.zeros(B, 2, D_FEAT))


class TestTargetsFromSequence(unittest.TestCase):
    def test_slices_offsets_and_masks_the_end_of_the_trajectory(self):
        T = 6
        feats = torch.arange(T, dtype=torch.float32)[None, :, None].expand(2, T, D_FEAT).clone()
        targets, valid = targets_from_sequence(feats, t=torch.tensor([1, 4]), offsets=[1, 3])
        self.assertEqual(targets.shape, (2, 2, D_FEAT))
        self.assertEqual(valid.tolist(), [[True, True], [True, False]])
        torch.testing.assert_close(targets[0, :, 0], torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(targets[1, :, 0], torch.tensor([5.0, 0.0]))

    def test_scalar_t_and_negative_index(self):
        feats = torch.randn(3, 5, D_FEAT)
        targets, valid = targets_from_sequence(feats, t=0, offsets=[-1, 0, 4, 5])
        self.assertEqual(valid.tolist(), [[False, True, True, False]] * 3)
        torch.testing.assert_close(targets[:, 1], feats[:, 0])
        torch.testing.assert_close(targets[:, 2], feats[:, 4])
        self.assertTrue(torch.all(targets[:, 0] == 0) and torch.all(targets[:, 3] == 0))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            targets_from_sequence(torch.zeros(4, D_FEAT), 0, [1])
        with self.assertRaises(ValueError):
            targets_from_sequence(torch.zeros(3, 4, D_FEAT), [0, 1], [1])


class TestFutureFeaturePredictionHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.h = torch.randn(B, K, D)

    def test_output_shape_and_offset_pooling(self):
        head = FutureFeaturePredictionHead(D, D_FEAT, offsets=[1, K, 2 * K])
        self.assertEqual(head(self.h).shape, (B, 3, D_FEAT))
        pooled = head.pool(self.h)
        torch.testing.assert_close(pooled[:, 0], self.h[:, 0])
        torch.testing.assert_close(pooled[:, 1], self.h[:, K - 1])
        torch.testing.assert_close(pooled[:, 2], self.h[:, K - 1])  # clamped to the last query
        pred = head(self.h)
        self.assertFalse(torch.allclose(pred[:, 1], pred[:, 2]))  # offset embedding separates shared inputs

    def test_mean_pooling(self):
        head = FutureFeaturePredictionHead(D, D_FEAT, offsets=[1, 2], pooling="mean", mlp_hidden=8)
        pooled = head.pool(self.h)
        torch.testing.assert_close(pooled[:, 0], self.h.mean(dim=1))
        torch.testing.assert_close(pooled[:, 1], self.h.mean(dim=1))
        self.assertEqual(head(self.h).shape, (B, 2, D_FEAT))

    def test_loss_uses_configured_weights_and_backprops(self):
        head = FutureFeaturePredictionHead(D, D_FEAT, offsets=[1, K], cosine_weight=0.5, mse_weight=0.25)
        target = torch.randn(B, 2, D_FEAT)
        out = head.loss(head(self.h), target, torch.ones(B, 2, dtype=torch.bool))
        torch.testing.assert_close(out["loss"], 0.5 * out["cosine_loss"] + 0.25 * out["mse_loss"])
        out["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_invalid_construction(self):
        with self.assertRaises(ValueError):
            FutureFeaturePredictionHead(D, D_FEAT, offsets=[])
        with self.assertRaises(ValueError):
            FutureFeaturePredictionHead(D, D_FEAT, offsets=[1, 1])
        with self.assertRaises(ValueError):
            FutureFeaturePredictionHead(D, D_FEAT, offsets=[-1])
        with self.assertRaises(ValueError):
            FutureFeaturePredictionHead(D, D_FEAT, pooling="max")
        with self.assertRaises(ValueError):
            FutureFeaturePredictionHead(D, D_FEAT)(torch.zeros(B, D))


if __name__ == "__main__":
    unittest.main()
