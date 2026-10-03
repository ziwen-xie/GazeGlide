import unittest
import numpy as np
from gaze import GazeMap, TARGETS, head_features, head_adjustment

class GazeTests(unittest.TestCase):
    def test_mapping_and_clipping(self):
        targets = np.array(TARGETS)
        features = targets @ np.array([[.3, .04], [.02, .15]]) + [.01, -.02]
        model = GazeMap(features)
        for feature, target in zip(features, targets):
            np.testing.assert_allclose(model.predict(feature), target, atol=1e-8)
        np.testing.assert_allclose(model.predict([100,100]), [1,1])

    def test_reject_constant_or_confounded_gaze(self):
        for features in [np.ones((9,2)), np.column_stack((np.arange(9), np.arange(9)))]:
            with self.assertRaises(ValueError):
                GazeMap(features)

    def test_reject_inconsistent_samples(self):
        with self.assertRaises(ValueError):
            GazeMap(np.random.default_rng(12).normal(size=(9,2)))

    def test_head_reference_and_translation(self):
        reference = head_features(np.eye(4), [.5,.5], .3)
        np.testing.assert_allclose(head_adjustment(reference, reference), [0,0])
        moved = reference.copy()
        moved[2] += .03
        self.assertGreater(head_adjustment(moved, reference)[0], 0)
        np.testing.assert_allclose(head_adjustment(moved, reference, 0), [0,0])
        moved[0] = 1
        self.assertIsNone(head_adjustment(moved, reference))

    def test_pose_rotation(self):
        angle = .2
        rotation = np.eye(4)
        rotation[:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        pose = head_features(rotation, [.5,.5], .3)
        self.assertAlmostEqual(pose[0], angle)
        reference = head_features(np.eye(4), [.5,.5], .3)
        self.assertAlmostEqual(head_adjustment(pose, reference)[0], .04)
