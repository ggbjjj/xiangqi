"""Reproduce the imported run's figures from its unmodified local logs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
summary=json.loads((HERE/'training_summary.json').read_text(encoding='utf-8'))
run=PROJECT/summary['local_run_directory']
batches=[json.loads(line) for line in (run/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
events=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0})
events.Reload()

font_path=Path('C:/Windows/Fonts/msyh.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    font_name=font_manager.FontProperties(fname=str(font_path)).get_name()
else:
    font_name='Noto Sans CJK SC'
plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,
                     'font.size':11,'axes.labelcolor':'#53657b','text.color':'#19324a',
                     'xtick.color':'#60758a','ytick.color':'#60758a',
                     'svg.fonttype':'path','savefig.facecolor':'#f5f7fb'})

def rolling(values,window,weights=None):
    values=np.asarray(values,dtype=float)
    weights=np.ones_like(values) if weights is None else np.asarray(weights,dtype=float)
    top=np.r_[0.,np.cumsum(values*weights)]
    bottom=np.r_[0.,np.cumsum(weights)]
    end=np.arange(1,len(values)+1);begin=np.maximum(0,end-window)
    return (top[end]-top[begin])/(bottom[end]-bottom[begin])

def style(ax,title,xlabel,ylabel):
    ax.set_facecolor('white')
    ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=16)
    ax.set_xlabel(xlabel,labelpad=10)
    ax.set_ylabel(ylabel,labelpad=10)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    for name in ('left','bottom'):ax.spines[name].set_color('#d6dfeb')
    ax.grid(axis='y',color='#e9eef5',linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0,pad=7)

fig,axes=plt.subplots(2,2,figsize=(16,10.5),dpi=150)
fig.patch.set_facecolor('#f5f7fb')
fig.subplots_adjust(left=.075,right=.96,top=.815,bottom=.13,hspace=.49,wspace=.25)
fig.text(.075,.954,'Opus-5  /  第一轮古谱策略训练',fontsize=24,fontweight='bold')
fig.text(.075,.915,'完成 1 轮  ·  参数更新 195 次  ·  分段反传 2730 次  ·  价值头尚未训练',fontsize=13,color='#4c657d')
fig.text(.075,.877,'策略 / 价值 / PPO / 动作熵系数：1 / 0 / 0 / 0     ｜     T = 100，逻辑批量 = 128，微批量 = 64，梯度截断 = 16',
         fontsize=11,color='#667b90')

steps=np.array([b['global_step'] for b in batches])
loss=np.array([b['loss'] for b in batches])
counts=np.array([b['counts']['book'] for b in batches])
ax=axes[0,0]
style(ax,'策略损失随参数更新下降','参数更新次数（逻辑批次）','策略交叉熵 / 总损失')
ax.plot(steps,loss,color='#7ca7d2',linewidth=1,alpha=.6,label='原始批次损失')
ax.plot(steps,rolling(loss,11,counts),color='#246ca6',linewidth=2.2,label='11 批次滑动平均')
ax.scatter([steps[0],steps[-1]],[loss[0],loss[-1]],color='#246ca6',s=27,zorder=5)
ax.annotate(f'首批 {loss[0]:.3f}',(steps[0],loss[0]),xytext=(12,-12),textcoords='offset points',fontsize=10)
ax.annotate(f'末批 {loss[-1]:.3f}',(steps[-1],loss[-1]),xytext=(-90,15),textcoords='offset points',fontsize=10)
ax.legend(loc='upper right',frameon=False,fontsize=10)
ax.set_xlim(0,200)

segments=events.Scalars('loss/segment_total')
sx=np.array([point.step for point in segments]);sy=np.array([point.value for point in segments])
ax=axes[0,1]
style(ax,'每段反向传播的损失波动','反向传播段次数','分段损失')
ax.plot(sx,sy,color='#c3cfdf',linewidth=.65,alpha=.58,label='原始分段损失')
ax.plot(sx,rolling(sy,71),color='#347c75',linewidth=1.8,label='71 段滑动平均')
ax.legend(loc='upper right',frameon=False,fontsize=10)
ax.set_xlim(0,2730)

train=summary['train']['loss'];validation=summary['validation']['loss']
ax=axes[1,0]
style(ax,'第 1 轮：训练集与验证集损失','数据集','合法动作交叉熵')
bars=ax.bar(['训练集（整轮平均）','验证集（轮末）'],[train,validation],width=.48,
            color=['#347bbf','#d89749'],zorder=3)
ax.set_ylim(0,max(train,validation)*1.26)
for bar,value in zip(bars,[train,validation]):
    ax.text(bar.get_x()+bar.get_width()/2,value+.075,f'{value:.4f}',ha='center',fontsize=13,fontweight='bold')
ax.text(.02,.94,'只有 1 轮验证记录，未绘制跨轮趋势',transform=ax.transAxes,fontsize=10,color='#73859a')

ax=axes[1,1]
style(ax,'验证集动作命中率','预测方式','动作命中率')
accuracies=[summary['validation']['top1'],summary['validation']['top3']]
bars=ax.bar(['首选动作命中率','前三动作命中率'],accuracies,width=.48,color=['#288b7c','#8970b1'],zorder=3)
ax.set_ylim(0,.61);ax.yaxis.set_major_formatter(PercentFormatter(1))
for bar,value in zip(bars,accuracies):
    ax.text(bar.get_x()+bar.get_width()/2,value+.012,f'{value:.2%}',ha='center',fontsize=13,fontweight='bold')

fig.text(.075,.062,'验证集：1,440 局 / 108,001 个有效局面。策略损失与总损失重合；系数为零的损失未绘制。',
         fontsize=10,color='#667b90')
fig.text(.075,.031,f'运行：{summary["run_id"]}    ｜    平滑仅用于展示，评优使用原始验证集指标。',
         fontsize=9,color='#7c8da0')
for suffix in ('png','svg'):
    fig.savefig(HERE/f'training-curves.{suffix}',dpi=150)
plt.close(fig)

summary['chart_summary']={
    'first_batch_loss':float(loss[0]),'last_batch_loss':float(loss[-1]),
    'logical_batches':len(batches),'micro_batches':len(events.Scalars('loss/micro_total')),
    'backward_segments':len(segments),
    'training_elapsed_seconds':batches[-1]['progress']['elapsed_seconds'],
    'figures':['training-curves.png','training-curves.svg'],
}
(HERE/'training_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Saved:',HERE/'training-curves.png')
