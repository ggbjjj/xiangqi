"""Live TensorBoard scalars. Segment steps and optimizer steps are separate axes."""
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TrainingCurves:
    def __init__(self,directory,config):
        self.directory=Path(directory)/'tensorboard'
        self.writer=SummaryWriter(str(self.directory),flush_secs=1)
        self.weights=config.get('weights')
        (self.directory/'loss_weights.json').write_text(
            json.dumps({'weights':self.weights},ensure_ascii=False),encoding='utf-8')
        self.segments=0; self.micro_batches=0; self.logical_batches=0
        self.writer.add_text('说明/曲线与更新',
            'loss/segment_*：每个反向传播段（默认16步）的有效样本均值；横轴为段次数。\n\n'
            'loss/micro_*：完整微批；loss/logical_*：完整逻辑Batch；validation/*：独立古谱验证集。\n\n'
            'optimization/parameter_updates 记录实际参数更新次数；段数不是参数更新次数。\n\n'
            '中文窗口按系数展示总损失、策略损失、价值损失、PPO损失、动作熵损失；系数为0时隐藏。\n\n'
            '动作熵损失为负熵项 -λ_entropy H；原始分项标量仍保留未加权数值。\n\n'
            '训练损失和PPO损失不等于棋力；best按完整验证集的平均切片困惑度比较。',0)
        self.writer.add_text('配置',json.dumps(config,ensure_ascii=False,indent=2,default=str),0)
        self.closed=False

    def _terms(self,prefix,terms,counts,step):
        for key,value in terms.items():
            if counts.get(key,0) and (self.weights is None or self.weights.get(key,0)):
                self.writer.add_scalar(f'loss/{prefix}_{key}',value,step)

    def segment(self,event,*,epoch,global_step,progress=None):
        self.segments+=1
        self.writer.add_scalar('loss/segment_total',event['chunk_loss'],self.segments)
        self._terms('segment',event['chunk_terms'],event['chunk_counts'],self.segments)
        self.writer.add_scalar('optimization/epoch_at_segment',epoch,self.segments)
        self.writer.add_scalar('optimization/parameter_updates',global_step,self.segments)
        if event['finished_micro']:
            self.micro_batches+=1
            self.writer.add_scalar('loss/micro_total',event['loss'],self.micro_batches)
            self._terms('micro',event['terms'],event.get('counts',event['chunk_counts']),self.micro_batches)
        if progress is not None:
            for key in ('units_per_second','steps_per_second','games_per_second'):
                self.writer.add_scalar(f'speed/{key}',progress[key],self.segments)
            for key in ('utilization','memory_percent','memory_used_mib'):
                value=progress['gpu'].get(key)
                if value is not None: self.writer.add_scalar(f'gpu/{key}',value,self.segments)
        self.writer.flush()

    def logical_batch(self,report):
        self.logical_batches+=1; step=self.logical_batches
        self.writer.add_scalar('loss/logical_total',report['loss'],step)
        self._terms('logical',report['terms'],report['counts'],step)
        for key in ('grad_norm','ratio_mean','clip_fraction','approx_kl','micro_batches_per_second'):
            value=report.get(key)
            if value is not None: self.writer.add_scalar(f'optimization/{key}',value,step)
        self.writer.add_scalar('optimization/updates_after_batch',report['global_step'],step)
        self.writer.flush()

    def epoch(self,epoch,train,validation=None):
        self.writer.add_scalar('loss/epoch_total',train['loss'],epoch)
        if train['counts']['book'] and (self.weights is None or self.weights.get('book',0)):
            self.writer.add_scalar('book_ce/train',train['terms']['book'],epoch)
        if validation is not None:
            self.writer.add_scalar('book_ce/validation',validation['loss'],epoch)
            for key in ('loss','perplexity','corpus_perplexity','median_perplexity','segments','source_games','L','n'):
                self.writer.add_scalar(f'validation/{key}',validation[key],epoch)
        self.writer.flush()

    def best(self,epoch,parameter_id,loss):
        self.writer.add_scalar('selection/best_parameter_id',parameter_id,epoch)
        self.writer.add_scalar('selection/best_validation_perplexity',loss,epoch)
        self.writer.flush()

    def close(self):
        if not self.closed:
            self.closed=True; self.writer.flush(); self.writer.close()


def aggregate_reports(reports,weights):
    """Average each term over its own samples, including an uneven final Batch."""
    counts={k:sum(r['counts'][k] for r in reports) for k in ('book','ppo','value','entropy')}
    terms={k:sum(r['terms'][k]*r['counts'][k] for r in reports)/max(1,counts[k]) for k in counts}
    loss=sum((-1 if k=='entropy' else 1)*getattr(weights,k)*terms[k] for k in terms)
    return {'loss':loss,'terms':terms,'counts':counts}
