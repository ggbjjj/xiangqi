"""Read-only, incremental access to existing training events for the Chinese viewer."""
import math
import json
from pathlib import Path
import threading


def sample_points(points,limit=2000):
    if len(points)<=limit: return points
    return [points[i*(len(points)-1)//(limit-1)] for i in range(limit)]


class CurveReader:
    def __init__(self,logdir):
        self.logdir=Path(logdir).resolve(); self.cache={}; self.lock=threading.RLock()

    def runs(self):
        if not self.logdir.is_dir(): return []
        paths={p.parent.resolve() for p in self.logdir.rglob('events.out.tfevents.*') if p.is_file()}
        return sorted(p.relative_to(self.logdir).as_posix() for p in paths if p.is_relative_to(self.logdir))

    def read(self,run):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        if run not in self.runs(): raise ValueError('运行不存在，请重新选择')
        path=(self.logdir/run).resolve()
        with self.lock:
            if run not in self.cache:
                self.cache[run]=EventAccumulator(str(path),size_guidance={'scalars':0})
            accumulator=self.cache[run]; accumulator.Reload()
            series={}
            for tag in accumulator.Tags()['scalars']:
                points=[[e.step,e.value,e.wall_time] for e in accumulator.Scalars(tag) if math.isfinite(e.value)]
                series[tag]=sample_points(points)
        weights=None
        for candidate in (path/'loss_weights.json',path.parent/'config.json'):
            if not candidate.resolve().is_relative_to(self.logdir): continue
            try:
                supplied=json.loads(candidate.read_text(encoding='utf-8')).get('weights')
                if isinstance(supplied,dict):
                    weights={k:float(supplied.get(k,0)) for k in ('book','value','ppo','entropy')}
                    if any(not math.isfinite(v) or v<0 for v in weights.values()): weights=None
                    else: break
            except (OSError,ValueError,TypeError,AttributeError): pass
        validation=None
        candidate=path.parent/'validation'/'latest.json'
        if candidate.resolve().is_relative_to(self.logdir):
            try:
                validation=json.loads(candidate.read_text(encoding='utf-8'))
                if not isinstance(validation.get('histogram',{}).get('bins'),list): validation=None
            except (OSError,ValueError,AttributeError): validation=None
        selfplay=None
        candidate=path.parent/'selfplay.json'
        if candidate.resolve().is_relative_to(self.logdir):
            try:
                supplied=json.loads(candidate.read_text(encoding='utf-8'))
                if supplied.get('kind')=='selfplay': selfplay=supplied
            except (OSError,ValueError,AttributeError): pass
        return {'run':run,'kind':'selfplay' if selfplay is not None else 'training',
                'series':series,'weights':weights,'display_point_limit':2000,
                'validation_perplexity':validation,'selfplay':selfplay}
