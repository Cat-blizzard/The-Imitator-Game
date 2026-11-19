"""Nominal epochs for weighted sampling with replacement, not exhaustive passes."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EpochSchedule:
    episodes: int
    effective_batch: int

    def __post_init__(self):
        if min(self.episodes,self.effective_batch)<1:raise ValueError('Positive epoch sizes required')

    @property
    def updates(self):return math.ceil(self.episodes/self.effective_batch)

    def progress(self,step):
        return dict(epoch=step/self.updates,completed_epochs=step//self.updates,
                    updates_per_epoch=self.updates,effective_batch=self.effective_batch,
                    sampled_queries=step*self.effective_batch)

    def due(self,step,every):return every>0 and step>0 and step%(self.updates*every)==0

    def checkpoint_name(self,step):
        if step<=0 or step%self.updates:raise ValueError('Checkpoint must be on an epoch boundary')
        return f'epoch_{step//self.updates:04d}.pt'


class EpochMeter:
    """Aggregate every update, including updates omitted by the console logger."""
    def __init__(self,start_step=0,state=None):
        self.start_step=start_step;self.end_step=start_step;self.seconds=0.
        self.sums={};self.weights={};self.sources={}
        if state:self.__dict__.update(state)

    def add(self,step,metrics,sources,seconds):
        if step!=self.end_step+1:raise ValueError('Non-contiguous epoch metrics')
        self.end_step=step;self.seconds+=seconds
        for key,value in metrics.items():
            if key.startswith('action_count/'):continue
            weight=metrics.get('action_count/'+key.split('/',1)[1],1) if key.startswith('action_raw/') else 1
            self.sums[key]=self.sums.get(key,0.)+float(value)*weight
            self.weights[key]=self.weights.get(key,0.)+weight
        for key,value in sources.items():self.sources[key]=self.sources.get(key,0)+int(value)

    def state_dict(self):return dict(self.__dict__)

    def report(self,schedule):
        n=self.end_step-self.start_step;total=sum(self.sources.values())
        return dict(step=self.end_step,**schedule.progress(self.end_step),
                    start_step_exclusive=self.start_step,observed_updates=n,
                    partial_epoch=(n!=schedule.updates or self.start_step%schedule.updates!=0),
                    training_seconds=self.seconds,seconds_per_update=self.seconds/max(n,1),
                    metrics={k:v/max(self.weights[k],1) for k,v in self.sums.items()},
                    source_counts=dict(self.sources),
                    source_fractions={k:v/max(total,1) for k,v in self.sources.items()})
