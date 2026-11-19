from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from resimpair_grasp.common import binding, write
from resimpair_assets.semantic_scale_publish_v2 import validate_container_review


class IndependentContainerBindingTests(unittest.TestCase):
    def test_false_or_wrong_review_cannot_qualify_consumer(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); write(p / 'evidence.json', {'fixture': True})
            catalog = {'assets': [{'fixture': 'descriptor'}], 'affordance_index': binding(p / 'evidence.json')}
            write(p / 'catalog.json', catalog); write(p / 'other_catalog.json', catalog)
            review = {'strict_subset_and_metadata_integrity_passed': True, 'catalog': binding(p / 'catalog.json'),
                'affordance_index': catalog['affordance_index'], 'accepted_variants': 1,
                'semantics_unknown_preserved': True, 'reviewer': binding(p / 'evidence.json'),
                'independent_census': binding(p / 'evidence.json')}
            write(p / 'valid.json', review)
            self.assertTrue(validate_container_review(p / 'catalog.json', p / 'valid.json')['strict_subset_and_metadata_integrity_passed'])
            for i, change in enumerate([{'strict_subset_and_metadata_integrity_passed': False},
                    {'catalog': binding(p / 'other_catalog.json')}, {'accepted_variants': 2},
                    {'semantics_unknown_preserved': False}, {'affordance_index': None}]):
                bad = deepcopy(review); bad.update(change); write(p / f'bad_{i}.json', bad)
                with self.assertRaises(ValueError):
                    validate_container_review(p / 'catalog.json', p / f'bad_{i}.json')


if __name__ == '__main__':
    unittest.main()
