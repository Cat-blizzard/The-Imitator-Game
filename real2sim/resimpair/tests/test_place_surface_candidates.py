import unittest
import trimesh

from resimpair_tasks.place_surface_candidates_v1 import propose_patch


class PlaceSurfaceCandidatesTests(unittest.TestCase):
    def test_box_top_has_a_bound_upward_patch(self):
        mesh = trimesh.creation.box(extents=[.16, .16, .04])
        result = propose_patch(mesh)
        self.assertEqual(result['status'], 'upward_flat_patch_candidate')
        self.assertTrue(result['independent_support_check']['passed'])
        self.assertAlmostEqual(result['patch']['floor_z_m'], .02)
        self.assertEqual(result['physical_status'], 'unmeasured')

    def test_downward_single_sided_sheet_is_rejected(self):
        mesh = trimesh.Trimesh(vertices=[[-.08,-.08,0],[-.08,.08,0],[.08,.08,0],[.08,-.08,0]],
                               faces=[[0,1,2],[0,2,3]], process=False)
        self.assertEqual(propose_patch(mesh)['status'], 'no_patch_in_bounded_search')

    def test_small_sphere_does_not_claim_flat_support(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=.03)
        self.assertEqual(propose_patch(mesh)['status'], 'no_patch_in_bounded_search')


if __name__ == '__main__': unittest.main()
