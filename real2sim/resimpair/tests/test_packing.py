import unittest
import numpy as np
from resimpair_pickplace.packing import disk_layouts


class PackingTests(unittest.TestCase):
    def check_layouts(self, radii, radius, values):
        self.assertTrue(values)
        self.assertLessEqual(len(values),64)
        for xy in values:
            self.assertTrue(np.all(np.linalg.norm(xy,axis=1)+radii<=radius-.004+1e-9))
            for i in range(len(radii)):
                for j in range(i):
                    self.assertGreaterEqual(np.linalg.norm(xy[i]-xy[j])+1e-9,radii[i]+radii[j]+.012)

    def test_two_payloads_allocate_distinct_floor_locations(self):
        radii=np.array([.057,.039])
        self.check_layouts(radii,.127,disk_layouts(radii,.127))

    def test_full_container_rejects_instead_of_overlapping(self):
        self.assertEqual(disk_layouts([.06,.06],.10),[])
        self.assertEqual(disk_layouts([.12],.10),[])

    def test_more_than_two_objects_uses_bounded_search(self):
        radii=np.array([.025,.020,.019])
        self.check_layouts(radii,.12,disk_layouts(radii,.12,limit=8))

    def test_invalid_or_unbounded_request_is_rejected(self):
        for radii in [[],[.02]*7,[np.nan],[0.],[-.01]]:
            with self.assertRaises(ValueError):disk_layouts(radii,.1)
        with self.assertRaises(ValueError):disk_layouts([.02],.1,limit=10000)


if __name__=='__main__':unittest.main()
