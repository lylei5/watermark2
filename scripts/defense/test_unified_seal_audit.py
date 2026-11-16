"""Synthetic audit-contract checks; run on249 before real-result auditing."""
import csv
from pathlib import Path
import tempfile
import unittest

from audit_unified_seal_results import artifact, binomial, clean_budget, export, integer_threshold, sha


class SealAuditTests(unittest.TestCase):
    def test_budget_metadata_correction_preserves_original_evidence(self):
        receipt = dict(actual_total_unet_calls=102000, expected_total_unet_calls=204000,
                       full_budget_generations=1000, full_budget_inversions=1000, attacks=0)
        differences = clean_budget(receipt, 102000, 500)
        self.assertEqual(differences[0]['corrected'], 102000)
        self.assertEqual(receipt['expected_total_unet_calls'], 204000)

    def test_incomplete_measured_budget_is_rejected(self):
        receipt = dict(actual_total_unet_calls=101999, expected_total_unet_calls=102000,
                       full_budget_generations=1000, full_budget_inversions=1000, attacks=0)
        with self.assertRaises(ValueError):
            clean_budget(receipt, 101999, 500)

    def test_tied_boundary_is_conservative(self):
        self.assertEqual(integer_threshold([0]*95 + [3]*5, .04), 4)
        self.assertEqual(integer_threshold([0]*95 + [3]*5, .05), 1)

    def test_highest_count_and_zero_budget(self):
        self.assertEqual(integer_threshold([1024]*10, .01), 1025)
        self.assertEqual(integer_threshold([0]*10, .01), 1)

    def test_invalid_counts_rejected(self):
        for scores in [[], [float('nan')], [-1], [1025], [True]]:
            with self.assertRaises(ValueError):
                integer_threshold(scores, .01)

    def test_threshold_not_selected_from_validation(self):
        scores = [0]*90 + [2]*8 + [4]*2
        threshold = integer_threshold(scores, .05)
        self.assertEqual(threshold, 3)
        # Deliberately high validation acceptance must remain high.
        held_out = [4]*20
        self.assertEqual(sum(x >= threshold for x in held_out), 20)

    def test_exact_interval_boundaries(self):
        zero = binomial(0, 50)
        full = binomial(50, 50)
        self.assertEqual(zero['lower95'], 0.)
        self.assertEqual(full['upper95'], 1.)
        self.assertAlmostEqual(zero['upper95'], 1 - .025**(1/50))
        self.assertAlmostEqual(full['lower95'], .025**(1/50))

    def test_artifact_tampering_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            run = parent/'run'
            run.mkdir()
            inside = run/'record.json'
            inside.write_text('{}')
            digest = sha(inside)
            self.assertEqual(artifact(run, 'record.json', digest), inside.resolve())
            inside.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                artifact(run, 'record.json', digest)
            outside = parent/'outside.json'
            outside.write_text('{}')
            with self.assertRaises(ValueError):
                artifact(run, '../outside.json', sha(outside))

    def test_pending_results_are_not_zero_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            export(out, {})
            self.assertFalse((out/'main_table_seal.csv').exists())
            point = dict(target_fpr=.01, threshold=7, **binomial(18, 20000))
            export(out, dict(calibration=dict(working_points=[point])))
            with (out/'main_table_seal.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['legitimate_status'], 'pending')
            self.assertEqual(row['legitimate_accepts'], '')
            self.assertEqual(row['attack_status'], 'not_run')
            self.assertIn('--', (out/'main_table_seal.tex').read_text())

    def test_report_counts_remain_unconditional(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            point = dict(target_fpr=.001, threshold=9, **binomial(17, 20000))
            clean = dict(dataset='all', kind='genuine', working_point='natural_fpr_0.001', **binomial(420, 500))
            result = dict(calibration=dict(working_points=[point]), clean=dict(rows=[clean], diagnostics=[], costs={}))
            export(out, result)
            with (out/'main_table_seal.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['legitimate_sources'], '500')
            self.assertEqual(row['legitimate_accepts'], '420')
            self.assertEqual(row['legitimate_status'], 'verified')
            self.assertIn('420/500', (out/'main_table_seal.tex').read_text())
            self.assertIn(r'\%', (out/'main_table_seal.tex').read_text())


if __name__ == '__main__':
    unittest.main()
