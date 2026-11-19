#!/usr/bin/env python3
"""Bounded native RGB previews. Static actors only: not task executions."""
from pathlib import Path
import argparse,contextlib,fcntl,gc,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resimpair_runtime import ROOT,configure
configure()
from resimpair_grasp.common import binding,read,resolve,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=2);p.add_argument('--limit',type=int,default=12)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.limit<=32:raise ValueError('New output and finite preview limit required')
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    with contextlib.ExitStack() as stack:
        for path in resources.lock_paths(a.gpu):
            lock=stack.enter_context(path.open('r'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=resources.gpu_snapshot(a.gpu);resources.check_idle(snapshot)
        from examples.baselines.hypericl_task_expansion.common import bind_gpu
        gpu=bind_gpu(a.gpu)
        import gymnasium as gym
        import torch
        from PIL import Image,ImageDraw
        from mani_skill.utils.registration import REGISTERED_ENVS
        from corl2026_workshop.experiments.autonomous_pipeline_v4 import simulation,library
        from resimpair_grasp.scenes import item
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        a.output.mkdir(parents=True)
        write(a.output/'request.json',{'catalog':binding(a.catalog),'limit':a.limit,'gpu':snapshot,'api_calls':0,
              'task_executions':0,'purpose':'asset visual diagnostics; static actors, no manipulation or success inference'})
        rows=[]
        for i,b in enumerate(read(a.catalog)['assets'][:a.limit]):
            d=read(resolve(b));folder=a.output/f'{i:02d}';folder.mkdir();scene=library.make_scene('preview',0)
            scene['objects']=[item(b,d,'preview_object',[-.20,-.32],0.,False)];scene['supports']=[]
            h=d['half_size'][2]
            scene['cameras']={'opposite_high':{'eye':[.08,-.54,.25+h],'target':[-.20,-.32,h],
                               'width':384,'height':384,'fov':.78}}
            REGISTERED_ENVS.pop('ReSimPairSharedTable-v4',None);gym.envs.registration.registry.pop('ReSimPairSharedTable-v4',None)
            simulation.register(scene);t=time.perf_counter();env=None
            try:
                env=gym.make('ReSimPairSharedTable-v4',sim_backend='physx_cpu',render_backend=gpu['render_backend'],
                    control_mode='pd_joint_pos',obs_mode='rgb',reward_mode='none',render_mode='rgb_array',sensor_configs={'shader_pack':'rt-fast'})
                obs,_=env.reset(seed=0)
                rgb=simulation.array(obs['sensor_data']['opposite_high']['rgb']).reshape(384,384,3)
                image=folder/'preview.png';Image.fromarray(rgb).save(image)
                row={'index':i,'source_id':d['source_id'],'name':d['name'],'category':d['category'],
                     'descriptor':b,'image':binding(image),'camera':scene['cameras'],'elapsed_seconds':time.perf_counter()-t}
                rows.append(row);write(folder/'record.json',row);print(i,d['source_id'],flush=True)
            finally:
                if env is not None:env.close()
                gc.collect()
        write(a.output/'manifest.json',{'rows':rows,'task_executions':0,'api_calls':0})
        cols=4;tile=256;sheet=Image.new('RGB',(cols*tile,((len(rows)+cols-1)//cols)*(tile+30)),(245,245,245));draw=ImageDraw.Draw(sheet)
        for row in rows:
            i=row['index'];x=(i%cols)*tile;y=(i//cols)*(tile+30)
            im=Image.open(resolve(row['image']));im.thumbnail((tile,tile));sheet.paste(im,(x,y));draw.text((x+4,y+tile+4),f"{i:02d} {row['category']}",fill=(0,0,0))
        sheet.save(a.output/'contact_sheet.jpg')


if __name__=='__main__':main()
