"""Vectorized mesh antipodal search and exact-URDF gripper clearance.

No physics engine, robot trajectory, image generation or category whitelist.
"""
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from resimpair_grasp.common import ROOT,binding

# Optional acceleration is isolated from the retained pipeline's environment.
extra=ROOT/'runtime/tabletop_deps'
if extra.is_dir() and str(extra) not in sys.path:sys.path.insert(0,str(extra))


def transform(xyz=None,rpy=None):
    out=np.eye(4)
    if xyz is not None:out[:3,3]=np.asarray(xyz,float)
    if rpy is not None:out[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix()
    return out


def urdf_origin(element):
    if element is None:return np.eye(4)
    return transform(np.fromstring(element.get('xyz','0 0 0'),sep=' '),np.fromstring(element.get('rpy','0 0 0'),sep=' '))


def panda_meshes(opening=.08):
    """Native URDF collision elements, including fixed wrist camera attachments."""
    base=ROOT/'PolicyGen/mani_skill/assets/robots/panda';urdf=base/'panda_v3.urdf';root=ET.parse(urdf).getroot()
    links={x.attrib['name']:x for x in root.findall('link')};joints=root.findall('joint')
    frame={'panda_hand':transform([0,0,-.1034])};pending=['panda_hand'];meshes=[];bindings=[binding(urdf)]
    while pending:
        name=pending.pop()
        for i,col in enumerate(links[name].findall('collision')):
            geom=col.find('geometry');box=geom.find('box');file=geom.find('mesh')
            if box is not None:m=trimesh.creation.box(extents=np.fromstring(box.get('size'),sep=' '))
            elif file is not None:
                path=base/file.get('filename');m=trimesh.load(path,force='mesh',process=False);bindings.append(binding(path))
                m.apply_scale(np.fromstring(file.get('scale','1 1 1'),sep=' '))
            else:raise ValueError('Unsupported Panda collision primitive')
            m.apply_transform(frame[name]@urdf_origin(col.find('origin')))
            meshes.append((name,i,m))
        for j in joints:
            if j.find('parent').get('link')!=name:continue
            child=j.find('child').get('link');local=urdf_origin(j.find('origin'));kind=j.get('type')
            if kind=='prismatic':local=local@transform(np.fromstring(j.find('axis').get('xyz'),sep=' ')*opening/2)
            elif kind!='fixed':continue
            frame[child]=frame[name]@local;pending.append(child)
    return meshes,bindings


def bvh(mesh):
    from mplib.pymp.collision_detection import fcl
    obj=fcl.BVHModel();obj.beginModel(len(mesh.faces),len(mesh.vertices))
    obj.addSubModel(list(np.asarray(mesh.vertices,float)),list(np.asarray(mesh.faces,np.int32)));obj.endModel()
    return obj


class Clearance:
    """Conservative continuous open-hand approach sweep, not sampled body points."""
    def __init__(self,mesh,approach=.17):
        from mplib.pymp.collision_detection import fcl
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        self.fcl=fcl;self.geometry=bvh(mesh);self.obj=fcl.CollisionObject(self.geometry)
        edges=mesh.edges_unique
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(mesh.vertices),len(mesh.vertices)))
        _,labels=connected_components(graph,directed=False)
        self.component_points=mesh.vertices[np.unique(labels,return_index=True)[1]]
        self.mesh_ray=RayMeshIntersector(mesh);self.closed_mesh=mesh.is_watertight
        self.parts=[];self.vertices=[];self.gripper_bindings=[]
        for name,i,m in panda_meshes()[0]:
            # Convex swept volume contains each native part throughout straight
            # vertical descent. Over-conservatism can reject, not admit collision.
            vertices=np.vstack([m.vertices,m.vertices+[0,0,-approach]])
            swept=trimesh.convex.convex_hull(vertices);g=bvh(swept)
            equations=ConvexHull(swept.vertices).equations
            self.parts.append((g,fcl.CollisionObject(g),equations,swept.vertices[0]));self.vertices.append(m.vertices)
        self.vertices=np.vstack(self.vertices)

    def check(self,T,table_z,margin):
        from data_generation_methods.human_to_robot_pipeline.strict_motion import matrix_to_pose7
        vertices=self.vertices@T[:3,:3].T+T[:3,3]
        clearance=float(vertices[:,2].min()-table_z)
        if clearance<margin:return False,clearance,'gripper_table'
        pose=matrix_to_pose7(T)
        object_points_local=(self.component_points-T[:3,3])@T[:3,:3]
        part_points=[]
        for _,part,equations,point in self.parts:
            part.set_transformation(pose)
            if self.fcl.collide(part,self.obj,self.fcl.CollisionRequest()).is_collision():
                return False,clearance,'open_hand_or_approach_collision'
            if np.any(np.all(object_points_local@equations[:,:3].T+equations[:,3]<=1e-9,axis=1)):
                return False,clearance,'object_inside_swept_hand'
            part_points.append(point@T[:3,:3].T+T[:3,3])
        if self.closed_mesh and self.mesh_ray.contains_points(part_points).any():
            return False,clearance,'hand_inside_object'
        return True,clearance,None


def prepare_mesh(mesh,profile):
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():raise ValueError('empty_or_nonfinite_mesh')
    if len(mesh.faces)>profile['max_input_faces']:raise ValueError('input_face_budget')
    mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces,process=True,validate=True)
    mesh.remove_unreferenced_vertices();mesh.fix_normals(multibody=True)
    hull=mesh.convex_hull
    # Extents in the object's oriented box, not the arbitrary input axes.
    lengths=np.sort(hull.bounding_box_oriented.primitive.extents)
    if lengths[0]/lengths[-1]<profile['min_thickness_ratio']:raise ValueError('wide_flat_or_slender_skip')
    if lengths[0]<=1e-9:raise ValueError('degenerate_mesh')
    longest=float(np.clip(profile['target_middle_m']*lengths[-1]/lengths[1],profile['min_longest_m'],profile['max_longest_m']))
    scale=longest/lengths[-1];S=np.diag([scale,scale,scale,1.]);mesh.apply_scale(scale);hull.apply_scale(scale)
    if len(hull.faces)>profile['stable_hull_faces']:
        stable_hull=hull.simplify_quadric_decimation(face_count=profile['stable_hull_faces']).convex_hull
    else:stable_hull=hull
    poses,probabilities=trimesh.poses.compute_stable_poses(stable_hull,n_samples=1,sigma=0.,threshold=.015)
    if not len(poses):raise ValueError('no_geometric_support_pose')
    info={'original_obb_dimensions':lengths.tolist(),'scale_factor':float(scale),'target_longest_m':longest,
          'size_policy':'synthetic_gripper_scaled_before_grasp_search','real_metric_size_inferred':False,
          'center_of_mass_model':'uniform geometric convex support proxy','stable_pose_count':len(poses),
          'stable_hull_faces':len(stable_hull.faces),'mesh_faces':len(mesh.faces)}
    return mesh,S,list(zip(poses[:profile['stable_poses']],probabilities[:profile['stable_poses']])),info


def propose(mesh,profile,seed):
    """Horizontal opposing contacts, low-friction cones, full hand/table checks."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    start=time.perf_counter();rng=np.random.default_rng(seed)
    points,faces=trimesh.sample.sample_surface(mesh,profile['surface_samples'],seed=int(rng.integers(2**31)))
    normals=np.asarray(mesh.face_normals[faces]);horizontal=normals.copy();horizontal[:,2]=0
    norms=np.linalg.norm(horizontal,axis=1);cone=1/np.sqrt(1+profile['friction_coefficient']**2)
    valid=(norms>=cone)&(points[:,2]-mesh.bounds[0,2]>=profile['min_tcp_height_m'])
    p1=points[valid];n1=normals[valid];closing=-horizontal[valid]/norms[valid,None]
    if not len(p1):return [],{'rays':0,'elapsed_seconds':time.perf_counter()-start}
    ray=RayMeshIntersector(mesh);loc,ray_id,tri=ray.intersects_location(p1+closing*1e-5,closing,multiple_hits=False)
    if not len(loc):return [],{'rays':len(p1),'elapsed_seconds':time.perf_counter()-start}
    distance=np.linalg.norm(loc-p1[ray_id],axis=1);n2=mesh.face_normals[tri];dot=(n2*closing[ray_id]).sum(1)
    keep=(distance>=profile['min_width_m'])&(distance<=profile['max_width_m'])&(dot>=cone)
    pairs=[]
    centroid=mesh.centroid
    for p,q,n,c,w,cos2,normal2 in zip(p1[ray_id][keep],loc[keep],n1[ray_id][keep],closing[ray_id][keep],distance[keep],dot[keep],n2[keep]):
        center=(p+q)/2;depth=mesh.bounds[1,2]-center[2]
        cost=float(np.linalg.norm(center[:2]-centroid[:2])+.35*abs(depth-min(.026,mesh.extents[2]/2))+.08*w)
        pairs.append((cost,p,q,n,c,float(w),float(cos2),normal2))
    pairs.sort(key=lambda x:x[0]);clearance=Clearance(mesh,profile['approach_distance_m'])
    result=[];seen=set();rejected={};tested=0
    for cost,p,q,n,c,width,cos2,normal2 in pairs:
        center=(p+q)/2;angle=np.arctan2(c[1],c[0]);bucket=(*np.round(center/.005).astype(int),int(np.round(angle/.15)))
        if bucket in seen:continue
        seen.add(bucket);tested+=1
        T=np.eye(4);approach=np.array([0,0,-1.]);T[:3,:3]=np.column_stack([np.cross(c,approach),c,approach]);T[:3,3]=center
        ok,gap,reason=clearance.check(T,mesh.bounds[0,2],profile['table_clearance_m'])
        if not ok:rejected[reason]=rejected.get(reason,0)+1
        else:
            for flip in [False,True]:
                pose=T.copy()
                if flip:pose[:3,:2]*=-1
                # Wrist camera asymmetry requires checking the flipped hand too.
                if flip and not clearance.check(pose,mesh.bounds[0,2],profile['table_clearance_m'])[0]:continue
                result.append({'tcp_in_asset':pose.tolist(),'width_m':width,'contacts_in_asset':[p.tolist(),q.tolist()],
                    'contact_normals':[n.tolist(),normal2.tolist()],
                    'friction_cosines':[float(-n@c),cos2],'table_clearance_m':gap,'geometric_rank_cost':cost,
                    'approach_sweep_m':profile['approach_distance_m'],'flip':flip,'physical_status':'unmeasured'})
        if len(result)>=profile['max_grasps_per_pose'] or tested>=profile['max_clearance_tests']:break
    return result[:profile['max_grasps_per_pose']],{'rays':len(p1),'antipodal_pairs':len(pairs),'clearance_tests':tested,
        'rejections':rejected,'elapsed_seconds':time.perf_counter()-start,'ray_backend':'Embree'}
