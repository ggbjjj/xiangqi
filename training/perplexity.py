"""Persist all per-slice scores and a count-preserving interval plot."""
from bisect import bisect_right
import json
import math
from pathlib import Path


def perplexity_histogram(values):
    edges=[1,2,4,8,16,32,64,128,256,512,1024]
    counts=[0]*len(edges)
    for value in values:
        if not math.isfinite(value) or value<1: raise ValueError('困惑度必须为 >= 1 的有限数')
        counts[min(len(edges)-1,bisect_right(edges,value)-1)]+=1
    last=max((i for i,count in enumerate(counts) if count),default=0)
    bins=[]
    for i in range(max(3,last+1)):
        lower=edges[i]; upper=edges[i+1] if i+1<len(edges) else None
        bins.append({'lower':lower,'upper':upper,'count':counts[i],
                     'label':f'[{lower}, {upper})' if upper else f'≥ {lower}',
                     'percent':100*counts[i]/len(values) if values else 0.})
    return {'total':len(values),'bins':bins}


def _write_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def save_perplexity_report(report,directory,*,title='验证集困惑度',latest=None):
    """Return compact checkpoint metrics; raw rows remain in a separate JSONL."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    samples=report['samples']
    if len(samples)!=report['segments']: raise ValueError('每个验证切片必须恰有一个困惑度')
    histogram=perplexity_histogram([s['perplexity'] for s in samples])
    with (directory/'perplexities.jsonl').open('w',encoding='utf-8') as stream:
        for sample in samples: stream.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=None
    for candidate in (Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')):
        if candidate.is_file(): font=FontProperties(fname=str(candidate)); break
    with plt.rc_context({'axes.unicode_minus':False}):
        fig,ax=plt.subplots(figsize=(11,5.6),facecolor='#f3f6fa')
        ax.set_facecolor('white'); bins=histogram['bins']
        bars=ax.bar(range(len(bins)),[b['count'] for b in bins],color='#397cbb',width=.72)
        ax.set_xticks(range(len(bins)),[b['label'] for b in bins],fontproperties=font,fontsize=10)
        ax.set_ylabel('切片局数',fontproperties=font)
        ax.set_xlabel('困惑度区间（越低越好）',fontproperties=font,labelpad=12)
        ax.set_title(title,fontproperties=font,fontsize=17,pad=38,loc='left')
        note=(f'原局 {report["source_games"]:,} → 切片 {report["segments"]:,} 局  |  L={report["L"]}, n={report["n"]}'
              f'  |  平均 {report["perplexity"]:.3f}，中位数 {report["median_perplexity"]:.3f}')
        ax.text(0,1.035,note,transform=ax.transAxes,fontproperties=font,color='#5b7186',fontsize=10)
        ax.set_axisbelow(True); ax.grid(axis='y',color='#e6edf4')
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(0,max(1,max(b['count'] for b in bins))*1.22)
        for bar,b in zip(bars,bins):
            if b['count']: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+ax.get_ylim()[1]*.015,
                f'{b["count"]:,}\n{b["percent"]:.1f}%',ha='center',va='bottom',fontsize=10,color='#26425b')
        fig.text(.07,.025,'每个切片计数一次；填充步按正确动作概率 0.5 计入。',fontproperties=font,fontsize=10,color='#5b7186')
        fig.tight_layout(rect=(0,.06,1,1))
        for suffix in ('png','svg'): fig.savefig(directory/f'perplexity-histogram.{suffix}',dpi=180)
        plt.close(fig)
    compact={k:v for k,v in report.items() if k!='samples'}
    compact.update(histogram=histogram,artifacts={
        name:str((directory/file).resolve()) for name,file in (
            ('samples','perplexities.jsonl'),('png','perplexity-histogram.png'),('svg','perplexity-histogram.svg'))})
    _write_json(directory/'report.json',compact)
    if latest is not None:
        latest=Path(latest); latest.parent.mkdir(parents=True,exist_ok=True); _write_json(latest,compact)
    return compact
