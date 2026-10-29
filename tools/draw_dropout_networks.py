"""Draw the current HaiKu, Opus and fully expanded shared core as PNG + SVG.

Run from the repository root: python tools/draw_dropout_networks.py
Uses the project's existing Pygame drawing primitives; no Graphviz dependency.
"""
import html
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import draw_training_diagrams as drawing
from models.HaiKu.config import Haiku5Config
from models.Opus.config import OpusConfig

pygame=drawing.pygame
INK,MUTED=drawing.INK,drawing.MUTED
BLUE,BLUE_BG=drawing.BLUE,drawing.BLUE_BG
TEAL,TEAL_BG=drawing.TEAL,drawing.TEAL_BG
PURPLE,PURPLE_BG=drawing.PURPLE,drawing.PURPLE_BG
AMBER,AMBER_BG=drawing.AMBER,drawing.AMBER_BG
OUT=ROOT/'docs'/'diagrams'


class Figure(drawing.Figure):
    def text(self,x,y,value,*args,**kwargs):
        # Microsoft YaHei lacks several mathematical subscript/transpose glyphs.
        value=value.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉','0123456789')).replace('ᵀ','^T')
        return super().text(x,y,value,*args,**kwargs)

    def __init__(self,width,height,title,subtitle):
        self.width,self.height=width,height
        self.surface=pygame.Surface((width*drawing.SCALE,height*drawing.SCALE))
        self.surface.fill('white'); self.fonts={}
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width*2}" height="{height*2}" '
                  f'viewBox="0 0 {width} {height}" role="img">',
                  f'<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>',
                  f'<rect width="{width}" height="{height}" fill="white"/>']
        self.text(64,58,title,40,bold=True)
        self.text(66,110,subtitle,21,MUTED,max_width=width-132)

    def node(self,x,y,w,h,title,lines=(),color=BLUE,fill=BLUE_BG,size=24):
        self.rect(x,y,w,h,fill,color,radius=10)
        spacing=29; start=y+h/2-len(lines)*spacing/2
        self.text(x+w/2,start,title,size,color,True,'middle',w-20)
        for i,line in enumerate(lines):
            self.text(x+w/2,start+(i+1)*spacing,line,20,INK,anchor='middle',max_width=w-20)
        return (x+w/2,y+h)

    def arrow(self,points,color=BLUE,dashed=False): self.path(points,color,2.5,dashed,arrow=True)

    def footer(self,lines):
        y=self.height-35-30*(len(lines)-1)
        self.path([(64,y-27),(self.width-64,y-27)],'#D8E1E6',1)
        for i,line in enumerate(lines): self.text(66,y+i*30,line,19,MUTED,max_width=self.width-132)

    def save(self,stem):
        pygame.image.save(self.surface,str(OUT/f'{stem}.png'))
        (OUT/f'{stem}.svg').write_text('\n'.join(self.svg+['</svg>']),encoding='utf-8')
        print(OUT/f'{stem}.png')


def haiku():
    cfg=Haiku5Config()
    f=Figure(1800,1590,'HaiKu-5 / 加入残差 Dropout 后的完整网络',
             '默认 D=128，4 头，3 个独立核心；每核心 10 个 Dropout，全网 30 个，p=0.1')
    left,right,w=80,990,650
    steps=[(180,80,'局面输入 X_t',['[B,15,10,9]：红黑各 7 个棋子平面 + 当前走方']),
           (300,90,'Conv2d 15 → 64，3×3，padding=1',['输出 [B,64,10,9]']),
           (425,48,'GELU',[]),
           (510,90,'Conv2d 64 → 64，3×3，padding=1',['输出 [B,64,10,9]']),
           (635,48,'GELU',[]),
           (720,90,'Conv2d 64 → 128，1×1',['输出 [B,128,10,9]']),
           (850,85,'Flatten 空间维 + Transpose',['[B,128,10,9] → [B,90,128]']),
           (990,115,'内容与位置融合 Z_t',['LN_content(CNN 特征) + α × LN_position(R+C)',
                                         'R：Embedding(10,128)；C：Embedding(9,128)'])]
    for i,(y,h,title,lines) in enumerate(steps):
        f.node(left,y,w,h,title,lines)
        if i: f.arrow([(left+w/2,steps[i-1][0]+steps[i-1][1]),(left+w/2,y)])
    f.arrow([(left+w,1047),(875,1047),(875,228),(right,228)],TEAL)
    for i,y in enumerate((180,410,640)):
        f.node(right,y,w,155,f'独立核心 {i+1}  ·  10 个 Dropout',
               ['历史注意力 ×2 + 空间注意力 ×2 + 交叉注意力 ×1',
                f'4 头；D=128；FFN 128→512→128；KV 窗口 {cfg.context_plies}'],TEAL,TEAL_BG)
        if i: f.arrow([(right+w/2,y-75),(right+w/2,y)],TEAL)
    f.node(right,870,w,110,'策略头 Linear(128,90)',['每个起点预测 90 个终点 → [B,90,90]'])
    f.arrow([(right+w/2,795),(right+w/2,870)],TEAL)
    f.node(right,1045,w,115,'Flatten → 8100 动作 logits',['合法动作屏蔽 → 全动作 Softmax → 落子概率'])
    f.arrow([(right+w/2,980),(right+w/2,1045)])
    f.node(80,1220,1560,125,'每个 Block 的两条残差分支',
           ['h = LN₁(X + Dropout(GELU(AttentionOutput)))',
            '输出 = LN₂(h + Dropout(Linear₂(GELU(Linear₁(h)))))'],AMBER,AMBER_BG)
    f.footer(['每个核心独立 β、历史 KV 缓存及参数；共享 CNN 和位置编码只执行一次。详细逐层结构见子架构图。',
              '监督训练启用 Dropout；验证、对弈及含有效 PPO 的逻辑批量关闭。Dropout 无参数，旧权重可严格加载。',
              '依据 models/HaiKu/network.py 与 models/common.py；默认参数量仍为 3,043,614。'])
    f.save('haiku-5-dropout')


def opus():
    cfg=OpusConfig()
    f=Figure(2040,1850,'Opus-5 / 五核心策略与价值网络',
             '每核心 10 个 Dropout，全网 50 个，p=0.1；两次跨分支单头融合保持原有连接')
    f.node(430,175,1180,110,'局面输入 [B,15,10,9] → 共享 CNN',
           ['Conv 3×3:15→64 → GELU → Conv 3×3:64→64 → GELU → Conv 1×1:64→128'])
    f.node(430,335,1180,115,'重排 [B,90,128] → 内容/位置融合 Z_t',
           ['Z_t = LN_content(CNN 特征) + α × LN_position(RowEmbedding + ColEmbedding)'])
    f.arrow([(1020,285),(1020,335)])
    left,right,w=105,1155,780
    f.arrow([(1020,450),(1020,485),(495,485),(495,530)],TEAL)
    f.arrow([(1020,485),(1545,485),(1545,530)],PURPLE)
    f.text(145,495,'策略分支',23,TEAL,True)
    f.text(1195,495,'价值分支',23,PURPLE,True)
    for i,(y,number) in enumerate(((530,1),(760,2),(1220,3))):
        idx=number-1
        f.node(left,y,w,140,f'核心 {number} → F{number}',
               [f'{cfg.heads[idx]} 头；KV 窗口 {cfg.context_windows[idx]}；10 个 Dropout',
                '历史 ×2 / 空间 ×2 / 交叉 ×1；每个核心参数独立'],TEAL,TEAL_BG)
        if i: f.arrow([(495,(530 if i==1 else 760)+140),(495,y)],TEAL)
    f.node(right,530,w,140,'核心 4 → F4',
           ['8 头；KV 窗口 20；10 个 Dropout','输入与核心 1 相同，均为 Z_t'],PURPLE,PURPLE_BG)
    f.node(right,760,w,155,'跨分支融合 A → P_t',
           ['Q = Linear(F4)；K,V = 各自 Linear(F2)',
            '单头 SDPA；只访问当前 90 个位置'],PURPLE,PURPLE_BG)
    f.arrow([(1545,670),(1545,760)],PURPLE)
    f.arrow([(885,830),(1155,830)],TEAL)
    f.text(1020,798,'F2 → K,V',21,TEAL,True,'middle')
    f.node(right,1000,w,140,'核心 5 → F5',
           ['4 头；KV 窗口 30；10 个 Dropout','输入 P_t；历史 ×2 / 空间 ×2 / 交叉 ×1'],PURPLE,PURPLE_BG)
    f.arrow([(1545,915),(1545,1000)],PURPLE)
    f.node(right,1220,w,155,'跨分支融合 B → O_t',
           ['Q = Linear(F5)；K,V = 各自 Linear(F3)',
            '单头 SDPA；输出 [B,90,128]'],PURPLE,PURPLE_BG)
    f.arrow([(1545,1140),(1545,1220)],PURPLE)
    f.arrow([(885,1290),(1155,1290)],TEAL)
    f.text(1020,1258,'F3 → K,V',21,TEAL,True,'middle')
    f.node(left,1445,w,175,'注意力策略头',
           ['Q = Linear(F3)；K = Linear(F3)',
            'QKᵀ / √128 → [B,90,90] → Flatten(8100)',
            '合法动作屏蔽 → 全动作 Softmax'],TEAL,TEAL_BG)
    f.arrow([(495,1360),(495,1445)],TEAL)
    f.node(right,1445,w,175,'价值头 · 当前走方胜 / 和 / 负',
           ['90 个位置均值池化 → LayerNorm(128)',
            'Linear(128,64) → GELU → Linear(64,3)',
            'Softmax → [P胜, P和, P负]'],PURPLE,PURPLE_BG)
    f.arrow([(1545,1375),(1545,1445)],PURPLE)
    f.footer(['所有核心使用同一种详细子架构；各核心的头数与 KV 窗口见图。内部 FFN 为 128→512→128。',
              '新增 Dropout 均位于核心内的注意力/FFN 残差分支。跨分支融合 A/B 无残差、FFN 或时间缓存。',
              '监督训练启用；验证、对弈及含有效 PPO 的逻辑批量关闭。依据 models/Opus/network.py 与 models/common.py。'])
    f.save('opus-5-dropout')


def expanded_block(f,x,y,w,title,tag,history=False,cross=False,color=BLUE,fill=BLUE_BG):
    """Expand every operation, with explicit bypasses for both residual adds."""
    height=850
    f.rect(x,y,w,height,'#FAFCFE',color,True,12)
    f.text(x+w/2,y+28,title,25,color,True,'middle',w-20)
    labels=[('输入残差 X',False),
            ('Q / K / V：三个独立 Linear(D,D)',False),
            ('按 H 头重排，d_head = D/H',False)]
    if history or cross:
        labels.append(('K,V 追加独立滑动 KV 缓存（窗口 W）',False))
    else: labels.append(('K,V 仅含当前棋盘的 90 个位置',False))
    labels += [('SDPA：softmax(QKᵀ/√d_head + mask)V',False),
               ('合并 H 头 → 输出 Linear(D,D)',False),
               ('GELU',False),
               (f'Dropout {tag}a · p=0.1',True),
               ('Add：X + 注意力分支',False),
               ('LayerNorm₁(D) → h',False),
               ('FFN：Linear(D,512)',False),
               ('GELU',False),
               ('FFN：Linear(512,D)',False),
               (f'Dropout {tag}b · p=0.1',True),
               ('Add：h + FFN 分支',False),
               ('LayerNorm₂(D)',False),
               ('无效输入局面输出置 0',False)]
    centers=[]; node_x=x+60; node_w=w-110
    for i,(label,drop) in enumerate(labels):
        top=y+61+i*45; centers.append(top+17)
        if i: f.arrow([(node_x+node_w/2,top-11),(node_x+node_w/2,top)],color)
        f.node(node_x,top,node_w,34,label,color=AMBER if drop else color,
               fill=AMBER_BG if drop else fill,size=18)
    # Bypass tracks stay outside the operation boxes and enter the Add nodes.
    for start,stop,track in ((0,8,x+19),(9,14,x+39)):
        f.arrow([(node_x,centers[start]),(track,centers[start]),
                 (track,centers[stop]),(node_x,centers[stop])],color)
    return x+w/2,y+61+16*45+34


def core():
    f=Figure(2420,2510,'共享子架构 / 历史编码器—空间注意力—交叉融合（逐层展开）',
             'HaiKu 与 Opus 共用 Core；D=128，FFN=512；H/W 由所属核心决定；橙色为新增的 10 个 Dropout')
    f.node(80,160,1440,85,'核心输入 X_t  ·  [B,90,D]',['来自 CNN 融合特征、前一核心或 Opus 融合 A'],TEAL,TEAL_BG)
    f.node(1640,160,690,85,'绝对时间编码 e_t  ·  [1,1,D]',['sin/cos；可学习标量 β 为本核心独有'],PURPLE,PURPLE_BG)
    f.node(90,320,700,95,'历史入口 E₀ = LN_mix(X_t + βe_t)',
           ['LayerNorm(D)，带仿射参数'],BLUE,BLUE_BG)
    f.node(850,320,700,95,'空间入口 S₀ = X_t',
           ['保留当前空间信息'],TEAL,TEAL_BG)
    f.arrow([(440,245),(440,320)],BLUE)
    f.arrow([(1200,245),(1200,320)],TEAL)
    f.arrow([(1985,245),(1985,277),(775,277),(775,320)],PURPLE)
    f.arrow([(440,415),(440,455)],BLUE)
    e1=expanded_block(f,90,455,700,'历史 Block 1  ·  KV_self,1','H1',True)
    f.arrow([e1,(440,1345)],BLUE)
    e2=expanded_block(f,90,1345,700,'历史 Block 2  ·  KV_self,2','H2',True)
    f.arrow([(1200,415),(1200,455)],TEAL)
    s1=expanded_block(f,850,455,700,'空间 Block 1  ·  当前局面','S1',color=TEAL,fill=TEAL_BG)
    f.arrow([s1,(1200,1345)],TEAL)
    s2=expanded_block(f,850,1345,700,'空间 Block 2  ·  当前局面','S2',color=TEAL,fill=TEAL_BG)
    f.node(1640,365,690,135,'交叉查询准备 · 两次 LayerNorm',
           ['S₂ → LN_qcontent(D)，无仿射参数',
            '加 βe_t → LN_qmix(D)，带仿射参数'],PURPLE,PURPLE_BG)
    # Route completed lane outputs through the gutters to the cross inputs.
    f.arrow([s2,(1200,2242),(1588,2242),(1588,430),(1640,430)],TEAL)
    f.text(1550,2273,'S₂',20,TEAL,True)
    f.arrow([(1985,277),(2360,277),(2360,430),(2330,430)],PURPLE)
    f.node(1640,560,690,110,'交叉 Block 的三个来源',
           ['Q 输入 = 查询准备结果；K,V 输入 = E₂',
            '第一条残差的输入 X = S₂'],PURPLE,PURPLE_BG)
    f.arrow([(1985,500),(1985,560)],PURPLE)
    f.arrow([e2,(440,2295),(1610,2295),(1610,615),(1640,615)],BLUE)
    f.text(1470,2325,'E₂ → K,V',20,BLUE,True)
    f.arrow([(1985,670),(1985,725)],PURPLE)
    out=expanded_block(f,1640,725,690,'交叉 Block  ·  KV_cross','C',cross=True,
                       color=PURPLE,fill=PURPLE_BG)
    f.node(1640,1650,690,90,'核心输出 F_t  ·  [B,90,D]',
           ['传给下一核心、策略头或 Opus 跨分支融合'],PURPLE,PURPLE_BG)
    f.arrow([out,(1985,1650)],PURPLE)
    f.node(1640,1800,690,150,'数量核对 · 本核心共 10 层',
           ['历史 2×2 = 4；空间 2×2 = 4；交叉 1×2 = 2',
            '每个 Block：注意力分支 1 个 + FFN 分支 1 个',
            'Dropout 无可训练参数，也无 state_dict 条目'],AMBER,AMBER_BG)
    f.node(1640,2010,690,150,'掩码与缓存',
           ['历史 / 交叉：窗口最多 W 个局面（含当前）',
            '空间：只看当前 90 个位置；均非 token 下三角',
            'KV 缓存按序列独立；填充 key 屏蔽'],BLUE,BLUE_BG)
    f.footer(['注意力分支：h = LN₁(X + Dropout(GELU(Y)))；FFN 分支：LN₂(h + Dropout(Linear₂(GELU(Linear₁(h)))))。',
              'Y 为多头注意力合并并经过输出 Linear 后的结果；图中细线为绕过分支的残差输入，保留原有 LayerNorm 顺序。',
              '默认 p=0.1，可用 --dropout 修改；train() 随机丢弃并按 1/(1−p) 缩放，eval() 为恒等映射（p=1 时训练分支全 0）。',
              '依据 models/common.py 的 Core / Block / Attention；Opus 外部桥接注意力与策略头不属于本核心。'])
    f.save('shared-core-dropout-detailed')


def main():
    pygame.font.init(); OUT.mkdir(parents=True,exist_ok=True)
    try: haiku(); opus(); core()
    finally: pygame.font.quit()


if __name__=='__main__': main()
