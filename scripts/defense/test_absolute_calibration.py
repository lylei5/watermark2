import unittest
import numpy as np
from absolute_calibration import calibrate_absolute


class AbsoluteCalibrationTest(unittest.TestCase):
    def test_base_rejections_stay_in_denominator(self):
        result = calibrate_absolute(np.arange(100), np.arange(100) < 80, .95)
        self.assertFalse(result['feasible'])
        self.assertEqual(result['required_count'], 95)

    def test_uses_absolute_required_count(self):
        result = calibrate_absolute(np.arange(100), np.arange(100) < 80, .75)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['threshold'], 5)
        self.assertEqual(result['calibration_count'], 75)

    def test_reports_ties_instead_of_randomly_breaking_them(self):
        result = calibrate_absolute(np.ones(20), np.ones(20, bool), .95)
        self.assertEqual(result['calibration_count'], 20)
        self.assertEqual(result['tie_overshoot'], 1)

    def test_invalid_scores_fail(self):
        with self.assertRaises(ValueError):
            calibrate_absolute([np.nan], [True], .95)


if __name__ == '__main__':
    unittest.main()
