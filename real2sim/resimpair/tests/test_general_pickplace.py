import unittest
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from resimpair_pickplace.containers import cavity
from resimpair_pickplace.evaluator import compare_triangle_mesh,footprint
from resimpair_pickplace.reachability import pose
from resimpair_pickplace.scenes import container_fits


def open_tray():
    floor=trimesh.creation.box([.28,.28,.012]);floor.apply_translation([0,0,.006]);parts=[floor]
    for axis in range(2):
        for sign in [-1,1]:
            extent=[.28,.28,.05];extent[axis]=.008
            m=trimesh.creation.box(extent);xyz=[0,0,.025];xyz[axis]=sign*.136
            m.apply_translation(xyz);parts.append(m)
    return trimesh.util.concatenate(parts)


class GeneralPickPlaceTests(unittest.TestCase):
    def test_open_cavity_and_closed_box_are_distinct(self):
        c=cavity(open_tray())
        self.assertGreater(c['flat_floor_radius_m'],.10)
        self.assertAlmostEqual(c['floor_z_m'],.012)
        with self.assertRaisesRegex(ValueError,'no_supported_open_cavity'):
            cavity(trimesh.creation.box([.28,.28,.05]))

    def test_bottomless_ring_not_supported_container(self):
        tray=open_tray()
        # A rim with a missing floor must not receive a floor-support certificate.
        pieces=tray.split(only_watertight=False)
        walls=[p for p in pieces if p.extents[2]>.02]
        with self.assertRaises(ValueError):cavity(trimesh.util.concatenate(walls))

    def test_center_inside_does_not_imply_whole_object_inside(self):
        m=trimesh.creation.box([.08,.08,.04]);c={'center_xy_m':[0,0],'floor_z_m':.012,'flat_floor_radius_m':.07}
        T=pose([0,0,.032]);self.assertTrue(footprint(m,T,np.eye(4),c)['passed'])
        T[0,3]=.03
        self.assertFalse(footprint(m,T,np.eye(4),c)['passed'])
        self.assertFalse(container_fits({'cavity':c},[.06,.06,.01]))

    def test_missing_container_wall_rejected_despite_same_vertex_cloud(self):
        mesh=open_tray()
        actual={'type':'PhysxCollisionShapeTriangleMesh','vertices':mesh.vertices,
                'triangles':mesh.faces,'scale':[1,1,1],'local_pose':np.eye(4)}
        self.assertTrue(compare_triangle_mesh(mesh,actual)['passed'])
        bad={**actual,'triangles':mesh.faces[:-2]}
        with self.assertRaisesRegex(ValueError,'triangle_difference'):compare_triangle_mesh(mesh,bad)
        with self.assertRaisesRegex(ValueError,'must_not_use_convex_hull'):
            compare_triangle_mesh(mesh,{**actual,'type':'PhysxCollisionShapeConvexMesh'})

    def test_container_yaw_transform_preserves_containment(self):
        mesh=trimesh.creation.box([.06,.03,.04]);c={'center_xy_m':[.015,-.01],'floor_z_m':.02,'flat_floor_radius_m':.06}
        C=pose([.1,.3,0],1.9);O=C@pose([.015,-.01,.04],.3)
        self.assertTrue(footprint(mesh,O,C,c)['passed'])

    def test_model_base_frame_is_not_native_tcp(self):
        from resimpair_pickplace.graspgenx import tcp_transform
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'hand.urdf'
            p.write_text('<robot name="hand"><joint name="rotate" type="fixed">'
                '<parent link="world"/><child link="panda_hand"/><origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/></joint>'
                '<joint name="tcp" type="fixed"><parent link="panda_hand"/><child link="panda_hand_tcp"/>'
                '<origin xyz="0 0 0.1034"/></joint></robot>')
            T=tcp_transform(p)
            np.testing.assert_allclose(T[:3,3],[0,0,.1034])
            np.testing.assert_allclose(T[:3,1],[-1,0,0],atol=1e-12)
            self.assertFalse(np.allclose(T,np.eye(4)))

    def test_source_reachable_does_not_admit_unreachable_goal(self):
        from resimpair_pickplace.reachability import ArmSelector
        from resimpair_grasp.common import write,binding
        from scipy.spatial.transform import Rotation
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.json';T=np.eye(4);T[:3,:3]=Rotation.from_euler('x',np.pi).as_matrix()
            write(path,{'candidates':[{'id':'test','tcp_in_asset':T.tolist()}]})
            scene={'objects':[{'id':'object','position':[-.2,-.35,.06],'yaw':0.,'half_size':[.03,.03,.03]}],
                   'supports':[],'table_top_z':0.,'grasp_caches':{'object':binding(path)}}
            operation={'object':'object','target':{'support':'table','offset_xy':[2.,2.],'yaw':0.}}
            result=ArmSelector([[-.6,-.35,0],[-.6,.35,0]],seed=5).select(scene,operation)
            self.assertFalse(result['passed'])
            self.assertEqual(result['reason'],'no_single_arm_source_and_destination_grasp')
            self.assertTrue(any(v['stage']=='grasp' and v['status']=='Success' for r in result['decisions'] for v in r['checks']))

    def test_composition_rejects_unregistered_roles_and_implicit_handover(self):
        from resimpair_pickplace.composition import compose
        scene={'id':'test','objects':[],'supports':[]}
        with self.assertRaisesRegex(ValueError,'Movable role'):compose(scene,'bad',[('missing','missing')])
        with self.assertRaisesRegex(ValueError,'regrasp/handover'):
            compose(scene,'bad',[('same','a'),('same','b')])


if __name__=='__main__':unittest.main()
