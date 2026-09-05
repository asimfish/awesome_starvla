"""CPU tests for KeyframeHead, soft labels / BCE, the NMS + cooldown write policy, memory and curriculum."""

import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from starvla_lab.heads import (
    EvidenceMemory,
    KeyframeHead,
    KeyframeWritePolicy,
    TeacherStudentCurriculum,
    keyframe_bce_loss,
    nms_1d,
    soft_keyframe_labels,
)

B, K, D = 2, 4, 16


class TestSoftKeyframeLabels(unittest.TestCase):
    def test_peak_position_and_gaussian_width(self):
        labels = soft_keyframe_labels([[3]], 8, sigma=1.0)
        self.assertEqual(labels.shape, (1, 8))
        self.assertEqual(labels[0].argmax().item(), 3)
        self.assertAlmostEqual(labels[0, 3].item(), 1.0, places=6)
        for sigma in (0.5, 1.0, 2.0):
            row = soft_keyframe_labels([[3]], 8, sigma=sigma)[0]
            expected = math.exp(-1.0 / (2.0 * sigma**2))
            self.assertAlmostEqual(row[2].item(), expected, places=5)
            self.assertAlmostEqual(row[4].item(), expected, places=5)
            self.assertAlmostEqual(row[5].item(), math.exp(-4.0 / (2.0 * sigma**2)), places=5)
        # wider sigma -> heavier shoulders
        self.assertGreater(soft_keyframe_labels([[3]], 8, 2.0)[0, 5], soft_keyframe_labels([[3]], 8, 1.0)[0, 5])
        self.assertTrue(torch.all(labels >= 0) and torch.all(labels <= 1))

    def test_multiple_events_take_the_max(self):
        row = soft_keyframe_labels([[1, 6]], 8, sigma=1.0)[0]
        self.assertAlmostEqual(row[1].item(), 1.0, places=6)
        self.assertAlmostEqual(row[6].item(), 1.0, places=6)
        self.assertAlmostEqual(row[3].item(), math.exp(-2.0), places=5)

    def test_out_of_range_events_are_dropped(self):
        labels = soft_keyframe_labels([[-1, 8, 20], [7], []], 8, sigma=1.0)
        self.assertTrue(torch.all(labels[0] == 0))
        self.assertTrue(torch.all(labels[2] == 0))
        self.assertAlmostEqual(labels[1, 7].item(), 1.0, places=6)
        self.assertAlmostEqual(labels[1, 6].item(), math.exp(-0.5), places=5)

    def test_zero_sigma_is_one_hot_and_inputs_may_be_arrays(self):
        labels = soft_keyframe_labels([np.array([2]), torch.tensor([0, 3])], 4, sigma=0.0)
        torch.testing.assert_close(labels, torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]))
        self.assertEqual(soft_keyframe_labels([], 4, 1.0).shape, (0, 4))


class TestKeyframeBCELoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(B, 6)
        self.labels = soft_keyframe_labels([[1], [4]], 6, sigma=1.0)

    def test_unmasked_matches_torch_reference(self):
        torch.testing.assert_close(keyframe_bce_loss(self.logits, self.labels), F.binary_cross_entropy_with_logits(self.logits, self.labels))

    def test_per_sample_mask(self):
        ref = F.binary_cross_entropy_with_logits(self.logits[:1], self.labels[:1])
        torch.testing.assert_close(keyframe_bce_loss(self.logits, self.labels, torch.tensor([True, False])), ref)
        garbage = self.logits.clone()
        garbage[1] = 50.0
        torch.testing.assert_close(keyframe_bce_loss(garbage, self.labels, torch.tensor([True, False])), ref)

    def test_per_step_mask_and_pos_weight(self):
        mask = torch.zeros(B, 6, dtype=torch.bool)
        mask[0, 0] = True
        elementwise = F.binary_cross_entropy_with_logits(self.logits, self.labels, reduction="none")
        torch.testing.assert_close(keyframe_bce_loss(self.logits, self.labels, mask), elementwise[0, 0])
        weighted = F.binary_cross_entropy_with_logits(self.logits, self.labels, pos_weight=torch.full((6,), 7.0))
        torch.testing.assert_close(keyframe_bce_loss(self.logits, self.labels, pos_weight=7.0), weighted)

    def test_all_masked_is_zero_without_nan(self):
        logits = self.logits.clone().requires_grad_(True)
        loss = keyframe_bce_loss(logits, self.labels, torch.zeros(B, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_confident_correct_logits_give_near_zero_loss(self):
        hard = soft_keyframe_labels([[1], [4]], 6, sigma=0.0)
        logits = torch.where(hard > 0.5, torch.full_like(hard, 30.0), torch.full_like(hard, -30.0))
        self.assertLess(keyframe_bce_loss(logits, hard).item(), 1e-6)
        with self.assertRaises(ValueError):
            keyframe_bce_loss(logits, hard[:, :3])


class TestNMS(unittest.TestCase):
    def test_threshold_only(self):
        self.assertEqual(nms_1d([0.1, 0.9, 0.2, 0.3, 0.95, 0.1], 0.5, 0), [1, 4])
        self.assertEqual(nms_1d([0.1, 0.2], 0.5, 0), [])

    def test_window_keeps_local_peaks(self):
        self.assertEqual(nms_1d([0.6, 0.9, 0.7, 0.1, 0.55, 0.8], 0.5, 1), [1, 5])
        plateau = [0.9, 0.85, 0.8, 0.75, 0.7]
        self.assertEqual(nms_1d(plateau, 0.5, 0), [0, 1, 2, 3, 4])
        self.assertEqual(nms_1d(plateau, 0.5, 1), [0, 2, 4])
        self.assertEqual(nms_1d(plateau, 0.5, 2), [0, 3])
        self.assertEqual(nms_1d([0.7, 0.7], 0.5, 1), [0])  # earlier index wins ties

    def test_tensor_input_and_bad_shape(self):
        self.assertEqual(nms_1d(torch.tensor([0.0, 1.0, 0.0]), 0.5, 1), [1])
        with self.assertRaises(ValueError):
            nms_1d(torch.zeros(2, 3), 0.5, 1)


class TestKeyframeWritePolicy(unittest.TestCase):
    def test_threshold_gives_absolute_steps(self):
        policy = KeyframeWritePolicy(threshold=0.5)
        self.assertEqual(policy.decide([0.1, 0.9, 0.2, 0.3, 0.95, 0.1], t0=100), [101, 104])
        self.assertEqual(policy.events, [101, 104])
        self.assertEqual(policy.decide(torch.tensor([0.2, 0.1]), t0=200), [])

    def test_nms_collapses_a_high_probability_segment(self):
        policy = KeyframeWritePolicy(threshold=0.5, nms_window=1)
        self.assertEqual(policy.decide([0.6, 0.9, 0.7, 0.1, 0.55, 0.8]), [1, 5])

    def test_cooldown_within_and_across_calls(self):
        policy = KeyframeWritePolicy(threshold=0.5, cooldown=3)
        self.assertEqual(policy.decide([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], t0=0), [0, 6])
        self.assertEqual(policy.decide([1.0, 0.0, 0.0, 0.0], t0=8), [])  # 8 - 6 < 3
        self.assertEqual(policy.decide([1.0, 0.0, 0.0, 0.0], t0=9), [9])  # 9 - 6 == 3
        self.assertEqual(policy.events, [0, 6, 9])
        policy.reset()
        self.assertEqual(policy.events, [])
        self.assertEqual(policy.decide([1.0, 0.0, 0.0, 0.0], t0=8), [8])

    def test_duplicates_are_dropped_even_without_cooldown(self):
        policy = KeyframeWritePolicy(threshold=0.5)
        self.assertEqual(policy.decide([1.0, 0.0], t0=0), [0])
        self.assertEqual(policy.decide([1.0, 0.0], t0=0), [])
        self.assertEqual(policy.decide([0.0, 1.0], t0=0), [1])

    def test_fifo_bound_forgets_the_oldest_event(self):
        policy = KeyframeWritePolicy(threshold=0.5, max_events=2)
        self.assertEqual(policy.decide([1.0, 1.0, 1.0]), [0, 1, 2])
        self.assertEqual(policy.events, [1, 2])
        # the evicted step no longer counts as a duplicate
        self.assertEqual(policy.decide([1.0], t0=0), [0])
        self.assertEqual(policy.events, [2, 0])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            KeyframeWritePolicy(threshold=1.5)
        with self.assertRaises(ValueError):
            KeyframeWritePolicy(cooldown=-1)
        with self.assertRaises(ValueError):
            KeyframeWritePolicy(max_events=0)


class TestEvidenceMemory(unittest.TestCase):
    def test_fifo_eviction(self):
        memory = EvidenceMemory(max_events=2)
        self.assertIsNone(memory.write(1, "a"))
        self.assertIsNone(memory.write(2, "b"))
        self.assertEqual(memory.write(3, "c"), (1, "a"))
        self.assertEqual(len(memory), 2)
        self.assertEqual(memory.timesteps, [2, 3])
        self.assertEqual(memory.images, ["b", "c"])
        self.assertEqual(memory.entries, [(2, "b"), (3, "c")])
        memory.clear()
        self.assertEqual(len(memory), 0)
        with self.assertRaises(ValueError):
            EvidenceMemory(0)


class TestTeacherStudentCurriculum(unittest.TestCase):
    def test_monotone_bounded_and_anchored(self):
        curriculum = TeacherStudentCurriculum(total_steps=100, warmup=20)
        probs = [curriculum.teacher_prob(step) for step in range(0, 160)]
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertTrue(all(a >= b for a, b in zip(probs, probs[1:])))
        self.assertEqual(probs[0], 1.0)
        self.assertEqual(probs[20], 1.0)
        self.assertAlmostEqual(probs[60], 0.5)
        self.assertEqual(probs[100], 0.0)
        self.assertEqual(probs[159], 0.0)
        self.assertEqual(curriculum(60), curriculum.teacher_prob(60))

    def test_explicit_transition_and_endpoints(self):
        curriculum = TeacherStudentCurriculum(total_steps=100, warmup=20, transition=40, start=0.8, end=0.2)
        self.assertAlmostEqual(curriculum.teacher_prob(20), 0.8)
        self.assertAlmostEqual(curriculum.teacher_prob(40), 0.5)
        self.assertAlmostEqual(curriculum.teacher_prob(60), 0.2)
        self.assertAlmostEqual(curriculum.teacher_prob(99), 0.2)
        step = TeacherStudentCurriculum(total_steps=10, warmup=5, transition=0)
        self.assertEqual((step.teacher_prob(4), step.teacher_prob(5)), (1.0, 0.0))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            TeacherStudentCurriculum(total_steps=10, warmup=-1)
        with self.assertRaises(ValueError):
            TeacherStudentCurriculum(total_steps=10, start=1.5)


class TestKeyframeHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.h = torch.randn(B, K, D)

    def test_default_horizon_is_the_chunk_length(self):
        head = KeyframeHead(D)
        logits = head(self.h)
        self.assertEqual(logits.shape, (B, K))
        probs = head.probabilities(self.h)
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
        torch.testing.assert_close(probs, torch.sigmoid(logits))

    def test_horizon_resampling(self):
        head = KeyframeHead(D, horizon=2, mlp_hidden=8)
        per_step = head.mlp(self.h).squeeze(-1)
        torch.testing.assert_close(head(self.h), per_step.view(B, 2, 2).mean(-1))
        self.assertEqual(KeyframeHead(D, horizon=8)(self.h).shape, (B, 8))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            KeyframeHead(D, horizon=0)
        with self.assertRaises(ValueError):
            KeyframeHead(D)(torch.zeros(B, D))


if __name__ == "__main__":
    unittest.main()
