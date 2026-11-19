"""Compare view profiles on the same real physical layouts and assets."""
import json
from pathlib import Path
import argparse

from .bootstrap import ROOT,sha,write
from ..hypericl_task_expansion.common import bind_gpu


def asset_pool(family):
    approved={r['uid']for r in json.loads((ROOT/'asset_visual_review_v1.json').read_text())['assets']
              if r['approved_for_interaction_test']}
    return [r for r in json.loads((ROOT/'asset_adapters_v2/catalog.json').read_text())['assets']
        if r.get('uid')in approved and family in r.get('categories',[])and r['adapter']['top_obb_width_fits_panda']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'framing_probe_v1')
    p.add_argument('--gpu',type=int,default=0);a=p.parse_args();gpu=bind_gpu(a.gpu)
    import gymnasium as gym
    import numpy as np
    from PIL import Image,ImageDraw
    from .rich_runtime_v2 import register,VisibilityRecorder,settling_equivalence
    register();a.root.mkdir(exist_ok=False);results=[];pictures=[]
    for family in ('bottle','cup'):
        for profile in ('legacy','close'):
            env=gym.make('HyperICLRichAssetRelocation-v2',asset_pool=asset_pool(family),family=family,
                view_profile=profile,data_split='qualification',sim_backend='physx_cpu',render_backend=gpu['render_backend'],
                control_mode='pd_joint_pos',obs_mode='rgb+segmentation',reward_mode='none',
                sensor_configs=dict(shader_pack='rt-fast'))
            try:
                for seed in range(42000000,42000003):
                    rec=VisibilityRecorder(env);rec.reset(seed,'pick_first_to_left');metrics=rec.image_metrics()
                    picture=a.root/f'{family}_{seed}_{profile}.png'
                    rgb=rec.obs['sensor_data']['native']['rgb'].cpu().numpy().reshape(512,512,3)
                    Image.fromarray(rgb).save(picture);pictures.append((family,seed,profile,picture))
                    results.append(dict(family=family,seed=seed,profile=profile,metrics={k:np.asarray(v).tolist()for k,v in metrics.items()},
                        initial_physics_sha256=rec.initial_physics_sha256,image=str(picture),image_sha256=sha(picture),
                        randomization=rec.unwrapped.randomization))
            finally:env.close()
    comparisons=[]
    for family in ('bottle','cup'):
        for seed in range(42000000,42000003):
            pair=[r for r in results if r['family']==family and r['seed']==seed]
            assert len({r['initial_physics_sha256']for r in pair})==1,'camera comparison changed physics'
            old=next(r for r in pair if r['profile']=='legacy');new=next(r for r in pair if r['profile']=='close')
            assert old['randomization']['wall_material']==new['randomization']['wall_material']
            comparisons.append(dict(family=family,seed=seed,legacy_visible_pixels=old['metrics']['object_pixels_224'],
                close_visible_pixels=new['metrics']['object_pixels_224'],
                close_all_objects_in_frame=all(new['metrics']['projected_entirely_in_frame'])))
    sheet=Image.new('RGB',(4*384,3*414),'white');draw=ImageDraw.Draw(sheet)
    ordered=sorted(pictures,key=lambda v:(v[1],v[0],v[2]))
    for i,(family,seed,profile,picture)in enumerate(ordered):
        x=i%4*384;y=i//4*414;sheet.paste(Image.open(picture).resize((384,384)),(x,y))
        draw.text((x+4,y+386),f'{family} {seed} {profile}',fill='black')
    sheet.save(a.root/'comparison.jpg',quality=95)
    report=dict(status='framing_measured_review_pending',views=results,comparisons=comparisons,
        native_settling_equivalence=settling_equivalence(),source_sha256=sha(__file__),
        qa_segmentations_not_model_inputs=True,source_resolution=[512,512],policy_resolution=[224,224])
    write(a.root/'report.json',report)
    print(json.dumps(dict(comparisons=comparisons,comparison=str(a.root/'comparison.jpg'))),flush=True)


if __name__=='__main__':main()
