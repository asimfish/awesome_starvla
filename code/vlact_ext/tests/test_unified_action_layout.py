import math
import unittest

import numpy as np

from vlact_ext.unified_action_layout import (
    AGILEX_BIMANUAL_JOINTS,
    DEFAULT_LAYOUTS,
    FRANKA_DELTA_EE_GRIPPER,
    UNIFIED_DIM,
    EmbodimentLayout,
    TransformedDataset,
    UnifiedActionLayout,
    UnifiedActionTransform,
)


def _one_based(indices):
    return sorted(i + 1 for i in indices)


class TestLayoutRegistry(unittest.TestCase):
    def setUp(self):
        self.layout = UnifiedActionLayout()

    def test_franka_7d_to_20d(self):
        rng = np.random.default_rng(0)
        action = rng.uniform(-1, 1, size=(8, 7)).astype(np.float32)
        unified, active, periodic = self.layout.to_unified(action, "franka")
        self.assertEqual(unified.shape, (8, UNIFIED_DIM))
        self.assertEqual(unified.dtype, np.float32)
        # paper numbering: dims 13-18 (delta EE) and 19 (shared gripper) are active
        self.assertEqual(_one_based(np.flatnonzero(active)), [13, 14, 15, 16, 17, 18, 19])
        self.assertFalse(periodic.any())
        np.testing.assert_array_equal(unified[:, 12:18], action[:, :6])
        np.testing.assert_array_equal(unified[:, 18], action[:, 6])
        self.assertTrue(np.all(unified[:, ~active] == 0))

    def test_agilex_14d_to_20d(self):
        rng = np.random.default_rng(1)
        action = rng.uniform(-math.pi, math.pi, size=(16, 14))
        unified, active, periodic = self.layout.to_unified(action, "agilex")
        self.assertEqual(_one_based(np.flatnonzero(active)), list(range(1, 13)) + [19, 20])
        self.assertEqual(_one_based(np.flatnonzero(periodic)), list(range(1, 13)))
        np.testing.assert_array_equal(unified[:, 0:6], action[:, 0:6])  # left joints
        np.testing.assert_array_equal(unified[:, 6:12], action[:, 6:12])  # right joints
        np.testing.assert_array_equal(unified[:, 18], action[:, 12])  # left gripper -> shared slot
        np.testing.assert_array_equal(unified[:, 19], action[:, 13])  # right gripper
        self.assertTrue(np.all(unified[:, 12:18] == 0))

    def test_shared_gripper_slot(self):
        franka_active = FRANKA_DELTA_EE_GRIPPER.active_mask()
        agilex_active = AGILEX_BIMANUAL_JOINTS.active_mask()
        self.assertTrue(franka_active[18] and agilex_active[18])
        self.assertEqual(int((franka_active & agilex_active).sum()), 1)

    def test_round_trip(self):
        rng = np.random.default_rng(2)
        for tag, dim in (("franka", 7), ("agilex", 14)):
            action = rng.normal(size=(5, dim))
            unified, _, _ = self.layout.to_unified(action, tag)
            np.testing.assert_array_equal(self.layout.from_unified(unified, tag), action)
        single = rng.normal(size=(14,))
        unified, _, _ = self.layout.to_unified(single, "agilex")
        self.assertEqual(unified.shape, (UNIFIED_DIM,))
        np.testing.assert_array_equal(self.layout.from_unified(unified, "agilex"), single)

    def test_unknown_tag_raises_clear_error(self):
        with self.assertRaises(KeyError) as ctx:
            self.layout.to_unified(np.zeros((2, 7)), "ur5")
        msg = str(ctx.exception)
        self.assertIn("ur5", msg)
        self.assertIn("agilex", msg)
        self.assertIn("franka", msg)
        with self.assertRaises(KeyError):
            self.layout.from_unified(np.zeros((2, UNIFIED_DIM)), None)

    def test_wrong_native_dim_raises(self):
        with self.assertRaises(ValueError):
            self.layout.to_unified(np.zeros((2, 14)), "franka")
        with self.assertRaises(ValueError):
            self.layout.from_unified(np.zeros((2, 7)), "franka")

    def test_new_embodiment_is_a_dict_entry(self):
        # UR5: 6 joints + gripper, mapped onto the left-arm joint block and the shared gripper
        ur5 = EmbodimentLayout(slots=tuple(range(0, 6)) + (18,), periodic=tuple(range(6)))
        layouts = dict(DEFAULT_LAYOUTS)
        layouts["ur5"] = ur5
        registry = UnifiedActionLayout(layouts)
        unified, active, periodic = registry.to_unified(np.arange(7, dtype=np.float64), "ur5")
        self.assertEqual(sorted(registry.tags), ["agilex", "franka", "ur5"])
        np.testing.assert_array_equal(unified[:6], np.arange(6))
        self.assertEqual(unified[18], 6)
        self.assertEqual(_one_based(np.flatnonzero(periodic)), [1, 2, 3, 4, 5, 6])
        # register() works too, and validation catches bad slots
        registry.register("ur5_alias", ur5)
        self.assertIn("ur5_alias", registry.tags)
        with self.assertRaises(ValueError):
            registry.register("bad", EmbodimentLayout(slots=(0, 25)))
        with self.assertRaises(ValueError):
            EmbodimentLayout(slots=(0, 0))
        with self.assertRaises(ValueError):
            EmbodimentLayout(slots=(0, 1), periodic=(5,))

    def test_from_config_with_presets_and_explicit_slots(self):
        cfg = {
            "unified_dim": 20,
            "layouts": {
                "franka": {"preset": "franka"},
                "new_embodiment": "agilex",
                "ur5": {"slots": list(range(6)) + [18], "periodic": list(range(6))},
            },
        }
        registry = UnifiedActionLayout.from_config(cfg)
        self.assertEqual(sorted(registry.tags), ["franka", "new_embodiment", "ur5"])
        self.assertEqual(registry.get("new_embodiment"), AGILEX_BIMANUAL_JOINTS)
        self.assertEqual(registry.get("ur5").native_dim, 7)
        self.assertEqual(sorted(UnifiedActionLayout.from_config({"layouts": None}).tags), ["agilex", "franka"])
        with self.assertRaises(KeyError):
            UnifiedActionLayout.from_config({"layouts": {"x": {"preset": "nope"}}})


class TestSampleTransform(unittest.TestCase):
    def _sample(self, tag, dim, T=8):
        rng = np.random.default_rng(3)
        return {
            "action": rng.uniform(-1, 1, size=(T, dim)).astype(np.float16),
            "image": ["img"],
            "lang": "pick up the cup",
            "robot_tag": tag,
            "state": rng.uniform(-1, 1, size=(1, dim)).astype(np.float16),
        }

    def test_rewrites_action_and_adds_masks(self):
        transform = UnifiedActionTransform()
        sample = self._sample("agilex", 14)
        out = transform(sample)
        self.assertEqual(out["action"].shape, (8, UNIFIED_DIM))
        self.assertEqual(out["action"].dtype, np.float16)
        self.assertEqual(out["action_mask"].shape, (8, UNIFIED_DIM))
        self.assertEqual(out["periodic_mask"].shape, (8, UNIFIED_DIM))
        self.assertEqual(out["action_mask"].dtype, np.bool_)
        self.assertTrue(np.all(out["action_mask"][:, :12]) and not np.any(out["action_mask"][:, 12:18]))
        self.assertTrue(np.all(out["periodic_mask"][:, :12]) and not np.any(out["periodic_mask"][:, 12:]))
        # untouched fields are preserved and the input dict is not mutated
        self.assertEqual(out["lang"], sample["lang"])
        self.assertEqual(out["robot_tag"], "agilex")
        self.assertEqual(sample["action"].shape, (8, 14))
        self.assertNotIn("action_mask", sample)

    def test_optional_data_side_wrap(self):
        sample = self._sample("agilex", 14)
        sample["action"] = sample["action"].astype(np.float64)
        sample["action"][:, 0] = 190 * math.pi / 180  # a left joint beyond +pi
        sample["action"][:, 12] = 0.7  # gripper must stay untouched
        out = UnifiedActionTransform(wrap_period=2 * math.pi)(sample)
        np.testing.assert_allclose(out["action"][:, 0], -170 * math.pi / 180)
        np.testing.assert_allclose(out["action"][:, 18], 0.7)
        # a per-slot period array is accepted too
        period = np.full(UNIFIED_DIM, 2.0)
        out2 = UnifiedActionTransform(wrap_period=period)(self._sample("franka", 7))
        self.assertEqual(out2["action"].shape, (8, UNIFIED_DIM))

    def test_unknown_tag_in_sample_raises(self):
        with self.assertRaises(KeyError):
            UnifiedActionTransform()(self._sample("ur5", 7))

    def test_transformed_dataset_proxy(self):
        class FakeDataset:
            tag = "franka"

            def __init__(self):
                self.items = [
                    {"action": np.zeros((4, 7), dtype=np.float32), "robot_tag": "franka", "lang": "a"},
                    {"action": np.zeros((4, 14), dtype=np.float32), "robot_tag": "agilex", "lang": "b"},
                ]

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                return dict(self.items[i])

            def save_dataset_statistics(self, path):
                return f"saved:{path}"

        ds = TransformedDataset(FakeDataset(), UnifiedActionTransform())
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]["action"].shape, (4, UNIFIED_DIM))
        self.assertEqual(ds[1]["action"].shape, (4, UNIFIED_DIM))
        self.assertEqual(ds.tag, "franka")
        self.assertEqual(ds.save_dataset_statistics("x"), "saved:x")
        # mixed embodiments now stack into one batch
        batch = np.stack([ds[0]["action"], ds[1]["action"]])
        self.assertEqual(batch.shape, (2, 4, UNIFIED_DIM))


if __name__ == "__main__":
    unittest.main()
