"""Read-only, incremental access to existing training events for the Chinese viewer."""
import math
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
        if run not in self.runs(): raise ValueError('训练运行不存在，请重新选择')
        path=(self.logdir/run).resolve()
        with self.lock:
            if run not in self.cache:
                self.cache[run]=EventAccumulator(str(path),size_guidance={'scalars':0})
            accumulator=self.cache[run]; accumulator.Reload()
            series={}
            for tag in accumulator.Tags()['scalars']:
                points=[[e.step,e.value,e.wall_time] for e in accumulator.Scalars(tag) if math.isfinite(e.value)]
                series[tag]=sample_points(points)
        return {'run':run,'series':series,'display_point_limit':2000}
