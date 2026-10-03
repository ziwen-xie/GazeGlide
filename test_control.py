import unittest
from control import ScrollControl

class ControlTests(unittest.TestCase):
    def test_dwell_direction_and_loss(self):
        c = ScrollControl(.3, .5, .7)
        self.assertEqual(c.step(.7, 1), 0)
        self.assertEqual(c.step(.7, 1.8), -1)
        self.assertEqual(c.step(.7, 1.85), 0)
        self.assertEqual(c.step(None, 2), 0)
        self.assertEqual(c.step(.7, 2.1), 0)
        self.assertEqual(c.step(.7, 3, False), 0)
        self.assertEqual(c.step(.3, 4), 0)
        self.assertEqual(c.step(.3, 4.8), 1)
        self.assertEqual(c.step(.5, 5), 0)

    def test_reversed_camera_axis(self):
        c = ScrollControl(.7, .5, .3)
        c.step(.3, 1)
        self.assertEqual(c.step(.3, 2), -1)

    def test_reject_bad_calibration(self):
        for values in [(.5, .5, .7), (.6, .5, .7)]:
            with self.assertRaises(ValueError):
                ScrollControl(*values)

if __name__ == '__main__':
    unittest.main()
