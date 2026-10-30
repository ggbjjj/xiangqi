"""Live outcome counts for the same Chinese dashboard used by training."""
import json
import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from .store import empty_outcomes


class SelfPlayCurves:
    def __init__(self,directory,config):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        config={**config,'kind':'selfplay'}
        (self.directory/'config.json').write_text(
            json.dumps(config,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
        self.writer=SummaryWriter(str(self.directory/'tensorboard'),flush_secs=1)
        self.history=(self.directory/'outcomes.jsonl').open('w',encoding='utf-8')
        self.state={**config,'status':'running','completed_games':0,'saved_games':0,'buffered_games':0,
                    'discarded_games':0,'unfinished_games':0,'outcomes':empty_outcomes(),
                    'saved_outcomes':empty_outcomes(),'discarded_outcomes':empty_outcomes()}
        self.closed=False
        self._record()

    def _record(self):
        step=self.state['completed_games']
        for scope,field in (('completed','outcomes'),('saved','saved_outcomes')):
            for key,value in self.state[field].items():
                self.writer.add_scalar(f'selfplay/{scope}_{key}',value,step)
        for key in ('completed_games','saved_games','buffered_games','discarded_games'):
            self.writer.add_scalar(f'selfplay/{key}',self.state[key],step)
        self.writer.flush()
        self.history.write(json.dumps(self.state,ensure_ascii=False,default=str)+'\n')
        self.history.flush()
        # The viewer never sees a partially written summary, including on Windows.
        target=self.directory/'selfplay.json'; temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
        os.replace(temporary,target)

    def game(self,row):
        self.state.update(completed_games=row['game'],saved_games=row['saved_games'],
                          buffered_games=row['buffered_games'],outcomes=row['outcomes'],
                          saved_outcomes=row['saved_outcomes'])
        self._record()

    def finish(self,report):
        for key in ('completed_games','saved_games','discarded_games','unfinished_games',
                    'outcomes','saved_outcomes','discarded_outcomes','generated_steps','inference_batches'):
            self.state[key]=report[key]
        self.state.update(status='interrupted' if report['interrupted'] else 'complete',buffered_games=0)
        self._record()
        (self.directory/'report.json').write_text(
            json.dumps(self.state,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

    def close(self):
        if not self.closed:
            try:
                if self.state['status']=='running':
                    self.state['status']='error'; self._record()
            finally:
                self.closed=True; self.history.close(); self.writer.close()
