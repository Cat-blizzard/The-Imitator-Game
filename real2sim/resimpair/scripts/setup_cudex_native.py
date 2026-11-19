#!/usr/bin/env python3
"""Recreate the optional native-Panda cuDex runtime without changing .venv.

Requires an existing compatible PyTorch, CUDA compiler and uv. Does not launch
GPU optimization, physical execution, training or API calls. See the pinned
upstream license before use. The separate campaign controls experiment budgets.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda-home',type=Path,default=Path('/usr/local/cuda-12.9'))
    p.add_argument('--arch',default='12.0')
    p.add_argument('--build-timeout',type=int,default=900)
    p.add_argument('--skip-install',action='store_true')
    a=p.parse_args()
    if not 1<=a.build_timeout<=900:raise ValueError('build timeout must be 1..900 seconds')
    metadata=json.loads((ROOT/'third_party/cudex_native/UPSTREAM.json').read_text())
    source=ROOT/'runtime/upstream/cudex-native-panda';deps=ROOT/'runtime/cudex_deps'
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone',metadata['repository'],str(source)],check=True)
        subprocess.run(['git','checkout','--detach',metadata['commit']],cwd=source,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if commit!=metadata['commit']:raise ValueError('unexpected upstream revision')
    patch=ROOT/'third_party/cudex_native/compatibility.patch'
    reverse=subprocess.run(['git','apply','--reverse','--check',str(patch)],cwd=source,capture_output=True)
    if reverse.returncode:
        subprocess.run(['git','apply','--check',str(patch)],cwd=source,check=True)
        subprocess.run(['git','apply',str(patch)],cwd=source,check=True)
    if not a.skip_install:
        uv=shutil.which('uv')
        if not uv:raise RuntimeError('uv is required for an isolated dependency target')
        subprocess.run([uv,'pip','install','--python',sys.executable,'--target',str(deps),'--no-deps',
            '-r',str(ROOT/'third_party/cudex_native/requirements.txt')],check=True,timeout=600)
    prefix=deps/'cmeel.prefix'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDA_HOME=str(a.cuda_home),
        TORCH_CUDA_ARCH_LIST=a.arch,MAX_JOBS='2',PYTHONPATH=str(deps),
        CUDEX_GEOMETRY_PREFIX=str(prefix),SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0')
    env['PATH']=str(deps/'bin')+os.pathsep+str(a.cuda_home/'bin')+os.pathsep+env['PATH']
    out=ROOT/'runtime/cudex_builds'/str(time.time_ns());out.mkdir(parents=True)
    start=time.monotonic();codes={}
    for name,where in [('geometry',source/'src/curobo/geom/cpp'),('cuda',source)]:
        with (out/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,'setup.py','build_ext','--inplace'],cwd=where,
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=max(1,a.build_timeout-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();code='timeout'
            codes[name]=code
        if code:break
    (out/'result.json').write_text(json.dumps({'codes':codes,'seconds':time.monotonic()-start,
        'commit':commit,'paid_calls':0,'gpu_executions':0},indent=2)+'\n')
    print(out)
    if any(codes.values()):raise SystemExit('Build failed; see retained logs. No automatic retries.')


if __name__=='__main__':main()
