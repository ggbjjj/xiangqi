"""补充课堂讲解图：双头协作、棋谱预训练、经验回放池。"""

import pygame

from draw_training_diagrams import (
    Figure, OUT, INK, MUTED, LINE, TEAL, TEAL_BG, BLUE, BLUE_BG,
    PURPLE, PURPLE_BG, AMBER, AMBER_BG,
)


def heads():
    f = Figure('策略头与价值头 · 如何共同决定走法', 'p 分配搜索方向，v 评价局面；MCTS 综合两者形成搜索后的走法分布 π')
    f.box(70, 195, 420, 180, '同一个局面 s', ['输入：15 × 10 × 9', '卷积与残差主干提取共享特征', '随后进入两个输出分支'])
    f.box(590, 170, 940, 110, '策略头：哪些动作值得优先搜索？',
          ['8100 个 logits → 合法掩码 + Softmax → p(a|s)',
           'p 是动作之间的概率分配，不是每个动作的获胜概率'], BLUE, BLUE_BG)
    f.box(590, 325, 940, 110, '价值头：这个局面对当前走方有多有利？',
          ['每个局面只输出一个 v(s) ∈ [−1, 1]，预测最终胜负得分',
           '评价候选走法时，要先试走到后续局面，再调用价值头'], PURPLE, PURPLE_BG)
    f.path([(490, 285), (540, 285), (540, 225), (590, 225)], BLUE, arrow=True)
    f.path([(540, 285), (540, 380), (590, 380)], PURPLE, arrow=True)
    f.text(70, 477, 'MCTS 的一次模拟；下面的过程重复多次', 24, TEAL, True)
    boxes = [
        (70, '① 选择分支', ['结合 p、平均价值 Q', '以及访问次数 N']),
        (450, '② 走到叶子局面', ['沿搜索树模拟走子', '尚未结束时调用网络']),
        (830, '③ 评价叶子', ['使用价值头预测 v', '终局直接使用胜负结果']),
        (1210, '④ 向上回传', ['双方视角逐层翻转符号', '累计价值与访问次数']),
    ]
    for x, title, lines in boxes:
        f.box(x, 515, 320, 108, title, lines, title_size=25, body_size=20)
    for start, end in [(390, 450), (770, 830), (1150, 1210)]:
        f.path([(start, 569), (end, 569)], TEAL, arrow=True)
    cols = [150, 490, 900, 1330]
    for x, label in zip(cols, ['走法', '搜索前先验 p', '搜索访问次数 N', '训练目标 π = N / ΣN']):
        f.text(x, 677, label, 21, INK, True, 'middle')
    f.path([(70, 700), (1530, 700)], LINE, 1)
    for y, row in zip([724, 758, 792], [('A', '55%', '40', '20%'), ('B', '30%', '140', '70%'), ('C', '15%', '20', '10%')]):
        for x, value in zip(cols, row):
            f.text(x, y, value, 22, BLUE if row[0] == 'B' else INK, anchor='middle')
    f.footer('表中为教学示例，非项目实测：200 次模拟后 B 被访问最多。网络先验 p 与搜索分布 π 可以不同；后者用于自对弈策略训练。')
    f.save('04_策略头价值头与搜索协作')


def warmup():
    f = Figure('棋谱预训练 · 样本从哪里来，网络学什么', '拟扩展方案：当前项目尚未实现棋谱导入与监督预训练模块；此图用于说明设计思路')
    f.box(70, 190, 420, 114, '① 整理完整棋谱', ['解析初始局面、走子序列与结果', '校验坐标、走方和每一步合法性'], AMBER, AMBER_BG, True)
    f.box(590, 190, 420, 114, '② 按整盘棋划分数据', ['训练集用于更新，验证集用于选模型', '避免同盘相似局面泄漏到两边'], AMBER, AMBER_BG, True)
    f.box(1110, 190, 420, 114, '③ 逐步还原对局', ['保存落子前局面 s 与高手动作 a*', '再执行该动作，进入下一步'], AMBER, AMBER_BG, True)
    f.path([(490, 247), (590, 247)], AMBER, dashed=True, arrow=True)
    f.path([(1010, 247), (1110, 247)], AMBER, dashed=True, arrow=True)
    f.path([(1320, 304), (1320, 353), (405, 353), (405, 395)], AMBER, dashed=True, arrow=True)
    f.path([(1175, 353), (1175, 395)], AMBER, dashed=True, arrow=True)
    f.box(70, 395, 670, 114, '策略标签：高手实际选择的动作 a*',
          ['把 a* 编成 8100 维 one-hot 标签', '策略损失：−log p(a*|s)；提高该动作概率'], BLUE, BLUE_BG)
    f.box(850, 395, 650, 114, '价值标签：该盘棋的最终结果 z',
          ['每个局面按“当前落子方”确定正负', '获胜 +1，失败 −1，和棋 0'], PURPLE, PURPLE_BG)
    f.text(800, 554, '示例：同一盘棋最终红胜', 25, INK, True, 'middle')
    f.text(800, 593, '红方落子前局面：z = +1        黑方落子前局面：z = −1', 25, PURPLE, anchor='middle')
    f.box(70, 661, 420, 115, '④ 组成小批量样本', ['每条数据为（s, a*, z）', '结果未知时只使用策略监督'], AMBER, AMBER_BG, True)
    f.box(590, 661, 420, 115, '⑤ 监督学习更新网络', ['L = CE(p, a*) + λ × (v − z)²', '反向传播，联合更新主干与双头'], AMBER, AMBER_BG, True, body_size=20)
    f.box(1110, 661, 420, 115, '⑥ 进入自对弈阶段', ['加载结构一致的预训练权重', '后续策略标签改由 MCTS 产生'], AMBER, AMBER_BG, True)
    f.path([(490, 719), (590, 719)], AMBER, dashed=True, arrow=True)
    f.path([(1010, 719), (1110, 719)], AMBER, dashed=True, arrow=True)
    f.footer('预训练目的是提供较好的初始参数；模仿高手不等于证明每步最优。切换到自对弈时保留网络参数，继续更新全部网络。')
    f.save('05_棋谱预训练样本与目标')


def replay():
    f = Figure('经验回放池 · 从一局棋到多轮网络更新', '当前实现：按时间保留最近样本；每轮收集新对局后，打乱整个池并进行小批量训练')
    f.box(70, 192, 420, 114, '① 当前网络自对弈', ['根节点加噪声，增加探索', '前 10 步采样，之后选访问最多的走法'], title_size=25, body_size=19)
    f.box(590, 192, 420, 114, '② 每步先保存（s, π）', ['s：落子前局面与当前走方', 'π：MCTS 访问次数归一化分布'])
    f.box(1110, 192, 420, 114, '③ 结束后补齐结果 z', ['按每个局面的落子方标注胜负', '组成完整经验（s, π, z）'])
    f.path([(490, 249), (590, 249)], TEAL, arrow=True)
    f.path([(1010, 249), (1110, 249)], TEAL, arrow=True)
    f.text(70, 362, '回放池：默认最多 30,000 条“局面样本”，容量满时自动移除最旧样本', 25, INK, True)
    f.path([(1320, 306), (1320, 390), (1365, 390), (1365, 427)], TEAL, arrow=True)
    for i, title in enumerate(['最旧样本', '历史对局样本', '历史对局样本', '本轮对局样本', '最新样本']):
        x = 100 + 285 * i
        f.box(x, 427, 250, 96, title, ['（s, π, z）'], TEAL if i >= 3 else MUTED,
              TEAL_BG if i >= 3 else '#F3F6F8', title_size=23, body_size=21)
    f.text(800, 565, '混合新旧对局以减少相邻局面的相关性；复用已有搜索数据，提高样本利用率', 23, MUTED, anchor='middle')
    f.path([(800, 523), (800, 542)], TEAL, arrow=True)
    f.path([(800, 586), (800, 610), (280, 610), (280, 656)], TEAL, arrow=True)
    f.box(70, 656, 420, 116, '④ 打乱整个回放池', ['每个 epoch 重新打乱样本顺序', '依次遍历全部小批量，默认 3 个 epoch'])
    f.box(590, 656, 420, 116, '⑤ 每批更新网络', ['默认每批 128 条，Adam 学习率 0.001', '共同优化策略头、价值头和共享主干'], body_size=19)
    f.box(1110, 656, 420, 116, '⑥ 新网络开始下一轮', ['使用更新后的参数生成新对局', '新经验加入池，旧经验逐步淘汰'])
    f.path([(490, 714), (590, 714)], TEAL, arrow=True)
    f.path([(1010, 714), (1110, 714)], TEAL, arrow=True)
    f.path([(1530, 714), (1565, 714), (1565, 155), (280, 155), (280, 192)], TEAL, arrow=True)
    f.text(800, 813, '回放池存在内存中，当前未保存到模型文件；结束程序后，池内经验不会自动恢复', 22, MUTED, anchor='middle')
    f.footer('依据：train.py。自对弈默认每轮 4 局、每步 600 次模拟；实际训练遍历整个池，不是优先经验回放。')
    f.save('06_经验回放池与迭代训练')


if __name__ == '__main__':
    pygame.font.init()
    OUT.mkdir(parents=True, exist_ok=True)
    heads()
    warmup()
    replay()
    pygame.font.quit()
