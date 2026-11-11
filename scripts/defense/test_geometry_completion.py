"""A partial or duplicated published trajectory must not enter scoring."""
import unittest
from score_frozen_geometry import validate_full_records,bind_frozen_alpha
import blind_geometry_verifier as geometry_verifier


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for task in range(4):
            for method in ['sembind','gs']:
                for kind in ['reference','cover']:
                    self.rows.append(dict(id=f'{task}_{method}_{kind}',task=task,method=method,kind=kind))
                for update in range(1,152,10):
                    self.rows.append(dict(id=f'{task}_{method}_{update}',task=task,method=method,kind='candidate',update=update))

    def test_complete_schedule(self):
        validate_full_records(self.rows)

    def test_missing_final_update(self):
        with self.assertRaises(AssertionError):validate_full_records(self.rows[:-1])

    def test_duplicate_update_with_distinct_id(self):
        self.rows[-1]['update']=141
        with self.assertRaises(AssertionError):validate_full_records(self.rows)

    def test_missing_reference_disguised_as_cover(self):
        self.rows[0]['kind']='cover'
        with self.assertRaises(AssertionError):validate_full_records(self.rows)

    def test_frozen_threshold_survives_library_roundoff(self):
        frozen=5.634806966794078e-12
        old=geometry_verifier.ALPHA
        try:
            self.assertEqual(bind_frozen_alpha(frozen,frozen*(1+1e-15)),frozen)
            self.assertEqual(geometry_verifier.ALPHA,frozen)
        finally:geometry_verifier.ALPHA=old

    def test_threshold_change_is_rejected(self):
        frozen=5.634806966794078e-12
        for computed in [frozen*1.01,0.,float('nan')]:
            with self.assertRaises(ValueError):bind_frozen_alpha(frozen,computed)


if __name__=='__main__':unittest.main()
