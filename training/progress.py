"""Live console progress and device telemetry; no extra GPU Python package required."""
from datetime import datetime,timedelta,timezone
import math
import os
import subprocess
import sys
import threading
import time

SHANGHAI=timezone(timedelta(hours=8))


def duration(seconds):
    if seconds is None or not math.isfinite(seconds): return '统计中'
    seconds=max(0,int(seconds)); hours,seconds=divmod(seconds,3600); minutes,seconds=divmod(seconds,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class GPUStats:
    def __init__(self,device,*,cuda=None,runner=None):
        self.device=str(device); self.runner=runner or subprocess.run; self.cuda=cuda
        self.index=None; self.identifier=None
        if not self.device.startswith('cuda'): return
        if self.cuda is None:
            import torch
            self.cuda=torch.cuda
        if not self.cuda.is_available(): return
        self.index=int(self.device.split(':')[1]) if ':' in self.device else self.cuda.current_device()
        props=self.cuda.get_device_properties(self.index)
        self.identifier=getattr(props,'uuid',None)
        if not self.identifier:
            visible=os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible:
                tokens=visible.split(',')
                if self.index<len(tokens): self.identifier=tokens[self.index].strip()

    def sample(self):
        result={'device':self.device,'utilization':None,'memory_used_mib':None,'memory_total_mib':None,
                'memory_percent':None,'allocated_mib':None,'reserved_mib':None}
        if self.index is None: return result
        try:
            free,total=self.cuda.mem_get_info(self.index)
            result.update(memory_used_mib=(total-free)/2**20,memory_total_mib=total/2**20,
                          allocated_mib=self.cuda.memory_allocated(self.index)/2**20,
                          reserved_mib=self.cuda.memory_reserved(self.index)/2**20)
        except (RuntimeError,ValueError): pass
        # Resolve the actual device UUID: CUDA_VISIBLE_DEVICES may remap logical GPU 0.
        if self.identifier:
            try:
                output=self.runner(['nvidia-smi','-i',str(self.identifier),
                    '--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,timeout=1.5,check=True,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0)
                values=[v.strip() for v in output.stdout.strip().splitlines()[0].split(',')]
                def number(v):
                    try: return float(v) if math.isfinite(float(v)) else None
                    except ValueError: return None
                util,used,total=map(number,values[:3])
                if util is not None and 0<=util<=100: result['utilization']=util
                if used is not None and total is not None and total>0:
                    result.update(memory_used_mib=used,memory_total_mib=total)
            except (OSError,subprocess.SubprocessError,ValueError,IndexError): pass
        if result['memory_total_mib']:
            result['memory_percent']=100*result['memory_used_mib']/result['memory_total_mib']
        return result


class ConsoleProgress:
    """Persistent terminal lines; plain periodic snapshots for redirected output."""
    def __init__(self,title,total,*,device='cpu',unit='微批',interval=1.,enabled=True,
                 fallback_work_total=None,stream=None,clock=None,wall_clock=None,provider=None):
        if interval<=0 or not math.isfinite(interval): raise ValueError('progress-interval 必须为正有限数')
        self.title,self.total,self.unit=title,total,unit
        self.interval,self.enabled=interval,enabled
        self.stream=stream or sys.stderr; self.clock=clock or time.perf_counter
        self.wall_clock=wall_clock or (lambda:datetime.now(timezone.utc))
        self.provider=provider or GPUStats(device)
        self.started=self.clock(); self.start_wall=self.wall_clock()
        self.completed=0.; self.work=0; self.inference_batches=0; self.games=0
        self.phase='准备中'; self.metrics={}; self.fallback_work_total=fallback_work_total
        self.status='运行中'; self.hardware={'device':str(device)}
        self.lock=threading.RLock(); self.stop=threading.Event(); self.thread=None; self.closed=False
        self.bars=[]

    def __enter__(self):
        if not self.enabled: return self
        if getattr(self.stream,'isatty',lambda:False)():
            from tqdm import tqdm
            self.bars=[tqdm(total=0,bar_format='{desc}',position=i,dynamic_ncols=True,
                            file=self.stream,leave=True) for i in range(6)]
        self.render()
        self.thread=threading.Thread(target=self._refresh,daemon=True,name='xiangqi-gpu-progress')
        self.thread.start(); return self

    def update(self,advance=0.,*,work=0,batches=0,games=0,phase=None,**metrics):
        with self.lock:
            self.completed=min(self.total,self.completed+advance); self.work+=work
            self.inference_batches+=batches; self.games+=games
            if phase is not None: self.phase=phase
            self.metrics.update(metrics)

    def snapshot(self):
        with self.lock:
            elapsed=max(0.,self.clock()-self.started)
            rate=self.completed/elapsed if elapsed>0 else 0.
            remaining=(max(0.,self.total-self.completed)/rate if rate>0 else None)
            estimate_source='实测速度'
            if remaining is None and self.work>0 and self.fallback_work_total:
                remaining=max(0,self.fallback_work_total-self.work)/(self.work/max(elapsed,1e-9))
                estimate_source='按局长上限估算'
            if self.status=='完成': remaining=0.
            finish=(self.wall_clock()+timedelta(seconds=remaining)).astimezone(SHANGHAI) if remaining is not None else None
            return {'completed':self.completed,'total':self.total,'elapsed_seconds':elapsed,
                    'remaining_seconds':remaining,'estimated_total_seconds':elapsed+remaining if remaining is not None else None,
                    'estimated_finish':finish.isoformat(timespec='seconds') if finish else None,
                    'estimate_source':estimate_source,'units_per_second':rate,
                    'steps_per_second':self.work/max(elapsed,1e-9),'games_per_second':self.games/max(elapsed,1e-9),
                    'inference_batches_per_second':self.inference_batches/max(elapsed,1e-9),
                    'phase':self.phase,'status':self.status,'metrics':dict(self.metrics),'gpu':dict(self.hardware)}

    def lines(self):
        s=self.snapshot(); fraction=min(1.,s['completed']/max(1,s['total']))
        fill=int(20*fraction); bar='#'*fill+'-'*(20-fill)
        finish=s['estimated_finish'].replace('T',' ') if s['estimated_finish'] else '统计中'
        lines=[f'{self.title} [{bar}] {fraction:5.1%} {s["completed"]:.1f}/{s["total"]} {self.unit} | {s["phase"]} | {s["status"]}',
               f'已用 {duration(s["elapsed_seconds"])} | 预计总时长 {duration(s["estimated_total_seconds"])} | 剩余 {duration(s["remaining_seconds"])}',
               f'预计结束 {finish} ({s["estimate_source"]})']
        if self.unit=='微批': speed=f'微批 {s["units_per_second"]:.2f}/s | 局 {s["games_per_second"]:.2f}/s'
        else: speed=f'对局 {s["units_per_second"]:.3f}/s | 推理微批 {s["inference_batches_per_second"]:.2f}/s'
        losses=' | '.join(f'{k}={s["metrics"][k]:.4f}' for k in ('L','棋谱','PPO','V','H') if k in s['metrics'])
        extra=' | '.join(f'{k}={v}' for k,v in s['metrics'].items() if k not in ('L','棋谱','PPO','V','H'))
        lines.append(f'{speed} | 单步 {s["steps_per_second"]:.1f}/s')
        lines.append(losses or extra or '等待首个微批量')
        gpu=s['gpu']; util=gpu.get('utilization'); memory=gpu.get('memory_percent')
        gpu_line=(f'GPU {gpu.get("device","N/A")} 利用率 {util:.0f}%' if util is not None else f'GPU {gpu.get("device","N/A")} 利用率 N/A')
        if memory is not None:
            gpu_line+=f' | 显存 {gpu["memory_used_mib"]/1024:.2f}/{gpu["memory_total_mib"]/1024:.2f} GiB ({memory:.1f}%)'
        else: gpu_line+=' | 显存 N/A'
        lines.append(gpu_line); return lines

    def render(self):
        if not self.enabled: return
        with self.lock:
            lines=self.lines()
            if self.bars:
                for bar,line in zip(self.bars,lines): bar.set_description_str(line,refresh=True)
            else:
                self.stream.write('\n'.join(lines)+'\n'); self.stream.flush()

    def _refresh(self):
        while not self.stop.is_set():
            try: hardware=self.provider.sample()
            except Exception: hardware={'device':getattr(self.provider,'device','N/A')}
            with self.lock: self.hardware=hardware
            if self.stop.wait(self.interval): break
            self.render()

    def write(self,text):
        with self.lock:
            if self.bars:
                from tqdm import tqdm
                tqdm.write(text,file=self.stream)
            else:
                self.stream.write(text+'\n'); self.stream.flush()

    def close(self,status='完成'):
        if self.closed: return
        self.closed=True; self.stop.set()
        if self.thread is not None: self.thread.join(timeout=2.)
        with self.lock:
            self.status=status; self.render()
            for bar in self.bars: bar.close()

    def __exit__(self,exc_type,*args): self.close('完成' if exc_type is None else '已停止')
