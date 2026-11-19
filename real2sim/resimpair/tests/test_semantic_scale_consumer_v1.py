import unittest
from resimpair_assets.semantic_scale_consumer_v1 import classify


def state(role='manipulation', metric=False):
    return {'role': role, 'size_confidence': 'source_metadata_geometry_agreement' if metric else 'engineering_category_or_morphology_assumption',
            'metric_metadata_status': 'matched_uniform_metric_metadata' if metric else 'no_metadata_uniform_shape_match'}


class SemanticScaleTests(unittest.TestCase):
    def test_whole_device_families_do_not_inherit_a_small_category_size(self):
        for name in ['Laptop', 'notebook computer', 'MacBook Pro', 'Desktop Computer', 'Old Television',
                     'Computer monitor from Poly by Google', 'Printer', 'TV Stand', 'Meuble TV',
                     'Bench (first Blender project)', 'Tool kit', 'Delivery truck']:
            result = classify({'name': name, 'source_categories': ['electronics_controls']}, state())
            self.assertTrue(result['automatic_manipulation_hold'], name)
            self.assertIsNone(result['recommended_size']['dimensions_m'])
            self.assertFalse(result['recommended_size']['inherit_legacy_normalized_parent'])

    def test_accessory_mentions_are_not_whole_device_assertions(self):
        for name in ['Universal TV Remote Control', 'Laptop box', 'Laptop sleeve', 'Computer mousepad',
                     'Notebook paper', 'Car Ride - Trash Can', 'Car wheel', 'Chair cushion', 'Camera']:
            self.assertFalse(classify({'name': name}, state())['automatic_manipulation_hold'], name)

    def test_toys_and_miniatures_remain_unknown_not_full_size(self):
        for name in ['Toy miniature chair', 'Mini Gaming Laptop', 'Diecast car', 'RC helicopter']:
            result = classify({'name': name}, state())
            self.assertEqual('unknown_miniature_or_toy', result['semantic_status'])
            self.assertTrue(result['potential_false_exclusion'])
            self.assertIsNone(result['recommended_size']['representative_length_m'])

    def test_original_non_pick_roles_are_not_promoted(self):
        result = classify({'name': 'Fridge', 'source_categories': ['appliances']}, state('background'))
        self.assertEqual('retain_existing_non_pick_role', result['consumer_action'])
        self.assertFalse(result['automatic_manipulation_hold'])

    def test_source_metric_geometry_evidence_is_distinct_from_inference(self):
        result = classify({'name': 'Toy laptop'}, state(metric=True))
        self.assertFalse(result['automatic_manipulation_hold'])
        self.assertEqual('retain_metric_grounded_candidate', result['consumer_action'])
        self.assertEqual('unknown_miniature_or_toy', result['semantic_status'])

    def test_tool_storage_not_automatically_multipart_toolset(self):
        self.assertFalse(classify({'name': 'Tool Box'}, state())['automatic_manipulation_hold'])
        self.assertTrue(classify({'name': 'Toolbox filled with tools'}, state())['automatic_manipulation_hold'])

    def test_no_source_specific_whitelists_or_id_dependency(self):
        a = classify({'name': 'Laptop', 'source_id': 'first'}, state())
        b = classify({'name': 'Laptop', 'source_id': 'different'}, state())
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
