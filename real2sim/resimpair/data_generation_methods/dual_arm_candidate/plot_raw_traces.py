"""Plot recorded object height/TCP motion without assigning success labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episode',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=np.load(args.episode/'arrays.npz',allow_pickle=False)
    obj=np.concatenate([a['object_states'],a['final_object_states'][None]])
    tcp=np.concatenate([a['tcp_world'],a['final_tcp_world'][None]])
    times=np.r_[a['timestamps_seconds'],a['timestamps_seconds'][-1]+.05]
    height=obj[:,:,2]-obj[0,:,2]
    speed=np.linalg.norm(np.diff(tcp[:,:,:3,3],axis=0),axis=-1)/np.diff(times)[:,None]
    centers=(times[:-1]+times[1:])/2
    both=(speed>.01).all(axis=1)
    plt.rcParams.update({'font.size':13,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),layout='constrained')
    for i,color in enumerate(['#b93033','#799b20']):
        axes[0].plot(times,height[:,i]*100,color=color,lw=2.3,label=['Object 0 (apple)','Object 1 (pear)'][i])
        axes[1].plot(centers,speed[:,i],color=color,lw=2.1,label=f'Arm {i}')
    axes[0].axhline(12,color='#555555',ls='--',lw=1,label='Frozen lift threshold')
    axes[0].set(xlabel='Simulation time (s)',ylabel='Recorded height gain (cm)',title='Both objects rise; full stability gate failed')
    axes[1].fill_between(centers,0,speed.max()*1.05,where=both,color='#4785c0',alpha=.14,label='Both TCP speeds > 0.01 m/s')
    axes[1].axhline(.01,color='#555555',ls='--',lw=1)
    axes[1].set(xlabel='Simulation time (s)',ylabel='TCP speed from recorded positions (m/s)',title=f'Same-clock motion overlap: {both.sum()*.05:.2f} s')
    for ax in axes:
        ax.set_xlim(0,times[-1]);ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=10)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
