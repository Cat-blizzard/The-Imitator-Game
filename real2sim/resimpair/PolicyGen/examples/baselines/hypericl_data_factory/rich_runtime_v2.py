"""Richer source video and explicit image quality; native physics stays in v1.

Two synchronized cameras share the same view: 224 RGB for existing policies,
512 RGB for references/future use. Segmentation is QA-only privileged data.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from .bootstrap import ROOT,sha,write
from . import rich_collect
from ..hypericl_task_expansion.collect import Recorder,array,require


def register():
    import sapien
    from mani_skill.utils.registration import register_env,REGISTERED_ENVS
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    rich_collect.register()
    parent=REGISTERED_ENVS['HyperICLRichAssetRelocation-v1'].cls

    @register_env('HyperICLRichAssetRelocation-v2',max_episode_steps=3000)
    class RichSourceEnv(parent):
        def __init__(self,*args,view_profile='close',data_split='train',**kwargs):
            self.view_profile=view_profile;self.data_split=data_split
            super().__init__(*args,**kwargs)

        def _load_scene(self,options):
            super()._load_scene(options)
            seed=int(options.get('layout_seed',0));rng=np.random.default_rng(seed+830003)
            close_fov=float(rng.uniform(1.18,1.30))
            if self.view_profile=='close':
                self.randomization['camera']['fov_radians']=close_fov
            elif self.view_profile=='balanced':
                self.randomization['camera']['fov_radians']=close_fov+.06
                self.randomization['camera']['eye']=(np.asarray(self.randomization['camera']['eye'])+[-.08,0,.07]).tolist()
            elif self.view_profile!='legacy':raise ValueError('unknown view profile')
            self.randomization['view_profile']=self.view_profile
            self.randomization['source_resolution']=[512,512]
            self.randomization['policy_resolution']=[224,224]
            choices=json.loads((ROOT/'background_inventory_v1/approved.json').read_text())['assets']
            split='train'if self.data_split=='qualification'else self.data_split
            choices=[r for r in choices if r['split']==split]
            material=choices[int(rng.integers(len(choices)))]
            if sha(material['path'])!=material['sha256']:raise ValueError('background source changed')
            render=sapien.render.RenderMaterial();render.set_base_color([1,1,1,1])
            render.set_base_color_texture(sapien.render.RenderTexture2D(material['path']));render.set_roughness(.9)
            b=self.scene.create_actor_builder()
            b.add_box_visual(half_size=[.00005,3,1.5],material=render)
            b.initial_pose=sapien.Pose([.7498,0,-.3]);b.build_static('appearance_only_wall')
            self.randomization['wall_material']=material
            self.randomization['wall_has_no_added_collision']=True
            for spec in self.asset_specs:
                for filename,digest in spec['files'].items():
                    if sha(filename)!=digest:raise ValueError('normalized asset changed')
            for key in ('diffuse_surface','hdri'):
                r=self.randomization[key]
                if sha(r['path'])!=r['sha256']:raise ValueError('lighting/material asset changed')

        @property
        def _default_sensor_configs(self):
            c=self.randomization['camera'];pose=sapien_utils.look_at(c['eye'],c['target'])
            return [CameraConfig('zed2i',pose,224,224,c['fov_radians'],.01,100),
                    CameraConfig('native',pose,512,512,c['fov_radians'],.01,100)]


class VisibilityRecorder(Recorder):
    def __init__(self,env):
        super().__init__(env)
        self.native_frames=[];self.visibility=[]

    def hold(self,steps,grippers=(1,1)):
        super().hold(steps,grippers)
        if not self.recording:
            consecutive=0
            for _ in range(120):
                stable=all(float(np.linalg.norm(array(o.linear_velocity)))<.01 and
                           float(np.linalg.norm(array(o.angular_velocity)))<.10 for o in self.unwrapped.objects)
                consecutive=consecutive+1 if stable else 0
                if consecutive>=10:break
                super().hold(1,grippers)
            require(consecutive>=10,'No physically stable initial pose within bounded settling budget')

    def reset(self,seed,program):
        super().reset(seed,program)
        camera=self.unwrapped._sensors['zed2i'].camera
        native=self.unwrapped._sensors['native'].camera
        self.extrinsic=array(camera.get_extrinsic_matrix()).reshape(3,4)
        self.intrinsic=array(camera.get_intrinsic_matrix()).reshape(3,3)
        self.native_intrinsic=array(native.get_intrinsic_matrix()).reshape(3,3)
        np.testing.assert_allclose(self.extrinsic,array(native.get_extrinsic_matrix()).reshape(3,4),atol=1e-6,rtol=0)
        self.ids=[int(array(o.per_scene_id).item())for o in self.unwrapped.objects]
        self.vertices=[np.asarray(o.get_first_collision_mesh(to_world_frame=False).convex_hull.vertices)
                       for o in self.unwrapped.objects]

    def image_metrics(self):
        sensor=self.obs['sensor_data'];seg=array(sensor['zed2i']['segmentation']).reshape(224,224)
        native_seg=array(sensor['native']['segmentation']).reshape(512,512)
        counts=[];counts_native=[];bounds=[];in_frame=[];visible_bounds=[]
        for i,obj in enumerate(self.unwrapped.objects):
            counts.append(int((seg==self.ids[i]).sum()));counts_native.append(int((native_seg==self.ids[i]).sum()))
            y,x=np.where(seg==self.ids[i])
            visible_bounds.append([int(x.min()),int(y.min()),int(x.max()),int(y.max())]if len(x)else[-1]*4)
            t=array(obj.pose.to_transformation_matrix()).reshape(4,4)
            world=self.vertices[i]@t[:3,:3].T+t[:3,3]
            camera=world@self.extrinsic[:,:3].T+self.extrinsic[:,3]
            if not np.all(camera[:,2]>.01):raise ValueError('object crosses camera near plane')
            pixels=camera@self.intrinsic.T;uv=pixels[:,:2]/pixels[:,2:3]
            box=np.r_[uv.min(0),uv.max(0)];bounds.append(box)
            in_frame.append(bool((box[:2]>=0).all()and(box[2:]<224).all()))
        return dict(object_pixels_224=counts,object_pixels_512=counts_native,
            projected_bounds_224=bounds,projected_entirely_in_frame=in_frame,visible_bounds_224=visible_bounds)

    def step(self,action):
        if self.recording:
            self.native_frames.append(array(self.obs['sensor_data']['native']['rgb']).reshape(512,512,3).copy())
            self.visibility.append(self.image_metrics())
        return super().step(action)


def settling_equivalence():
    def method(path,name):
        tree=ast.parse(Path(path).read_text())
        cls=next(n for n in ast.walk(tree)if isinstance(n,ast.ClassDef)and n.name==name)
        f=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='hold')
        return ast.dump(f,include_attributes=False)
    original=method(rich_collect.__file__,'SettledRecorder')
    current=method(__file__,'VisibilityRecorder')
    if original!=current:raise ValueError('settling behavior differs from the qualified native runtime')
    return hashlib.sha256(original.encode()).hexdigest()


def save_native(rec,folder):
    import imageio.v2 as imageio
    import av
    n=len(rec.rows);require(n==len(rec.native_frames)==len(rec.visibility),'camera/state length mismatch')
    video=folder/'video_native512.mp4'
    with imageio.get_writer(video,fps=rec.control_freq,codec='libx264',macro_block_size=16,
        ffmpeg_log_level='error',output_params=['-crf','18','-preset','fast','-tune','zerolatency',
        '-g',str(rec.control_freq),'-bf','0','-sc_threshold','0'])as writer:
        for frame in rec.native_frames:writer.append_data(frame)
    stamps=[]
    with av.open(str(video))as movie:
        for frame in movie.decode(video=0):
            require((frame.width,frame.height)==(512,512),'native resolution differs')
            stamps.append(float(frame.pts*frame.time_base))
    np.testing.assert_allclose(stamps,np.arange(n)/rec.control_freq,atol=1e-5,rtol=0)
    path=folder/'qa_visibility.npz'
    np.savez_compressed(path,object_pixels_224=np.asarray([v['object_pixels_224']for v in rec.visibility]),
        object_pixels_512=np.asarray([v['object_pixels_512']for v in rec.visibility]),
        projected_bounds_224=np.asarray([v['projected_bounds_224']for v in rec.visibility]),
        visible_bounds_224=np.asarray([v['visible_bounds_224']for v in rec.visibility]),
        projected_entirely_in_frame=np.asarray([v['projected_entirely_in_frame']for v in rec.visibility]),
        camera_from_world=rec.extrinsic,intrinsic_224=rec.intrinsic,intrinsic_512=rec.native_intrinsic)
    return dict(reference_video=dict(kind='file',path=str(video)),reference_resolution=[512,512],
        reference_video_sha256=sha(video),reference_native_fps=int(rec.control_freq),
        reference_raw_initial_rgb_sha256=hashlib.sha256(rec.native_frames[0].tobytes()).hexdigest(),
        qa_visibility=str(path),qa_visibility_sha256=sha(path),
        segmentation_is_QA_only_and_not_model_input=True,simultaneous_sensor_timestamps=True)
