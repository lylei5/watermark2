"""A partial or duplicated published trajectory must not enter scoring."""
import unittest
from score_frozen_geometry import validate_full_records


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


if __name__=='__main__':unittest.main()
