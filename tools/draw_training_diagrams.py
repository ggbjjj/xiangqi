"""生成可插入课件的网络、训练流程和残差块图；输出 PNG 与 SVG。"""

import html
import math
import os
from pathlib import Path

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '课件图片'
WIDTH, HEIGHT, SCALE = 1600, 900, 2
INK, MUTED, LINE = '#172D3A', '#566A75', '#9DADB6'
TEAL, TEAL_BG = '#137C75', '#EAF6F3'
BLUE, BLUE_BG = '#285AA8', '#EDF3FC'
PURPLE, PURPLE_BG = '#7751A7', '#F3EFFA'
AMBER, AMBER_BG = '#9B671D', '#FCF5E9'
REGULAR = 'C:/Windows/Fonts/msyh.ttc'
BOLD = 'C:/Windows/Fonts/msyhbd.ttc'


class Figure:
    def __init__(self, title, subtitle):
        self.surface = pygame.Surface((WIDTH * SCALE, HEIGHT * SCALE))
        self.surface.fill('white')
        self.fonts = {}
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="3200" height="1800" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img">',
            f'<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>',
            '<rect width="1600" height="900" fill="#FFFFFF"/>',
        ]
        self.text(68, 64, title, 42, bold=True)
        self.text(70, 116, subtitle, 22, MUTED)

    def point(self, point):
        return tuple(round(value * SCALE) for value in point)

    def text(self, x, y, value, size=24, color=INK, bold=False, anchor='start', max_width=None):
        key = size, bold
        if key not in self.fonts:
            self.fonts[key] = pygame.font.Font(BOLD if bold else REGULAR, size * SCALE)
        face = self.fonts[key]
        if max_width is not None and face.size(value)[0] > max_width * SCALE:
            raise ValueError(f'Text exceeds {max_width}px: {value}')
        rendered = face.render(value, True, color)
        rect = rendered.get_rect()
        if anchor == 'middle':
            rect.center = self.point((x, y))
        else:
            rect.midleft = self.point((x, y))
        self.surface.blit(rendered, rect)
        self.svg.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" '
            f'font-weight="{700 if bold else 400}" text-anchor="{anchor}" '
            f'dominant-baseline="central">{html.escape(value)}</text>')

    def rect(self, x, y, w, h, fill='white', stroke=LINE, dashed=False, radius=14):
        box = pygame.Rect(x * SCALE, y * SCALE, w * SCALE, h * SCALE)
        pygame.draw.rect(self.surface, fill, box, border_radius=radius * SCALE)
        if dashed:
            self.path([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
                      stroke, 1.5, dashed=True, svg=False)
        else:
            pygame.draw.rect(self.surface, stroke, box, width=2 * SCALE, border_radius=radius * SCALE)
        dash = ' stroke-dasharray="9 7"' if dashed else ''
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0 if dashed else radius}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')

    def path(self, points, color=LINE, width=2.5, dashed=False, arrow=False, svg=True):
        for start, end in zip(points, points[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if not length:
                continue
            if dashed:
                for offset in range(0, math.ceil(length), 17):
                    stop = min(offset + 10, length)
                    p1 = start[0] + dx * offset / length, start[1] + dy * offset / length
                    p2 = start[0] + dx * stop / length, start[1] + dy * stop / length
                    pygame.draw.line(self.surface, color, self.point(p1), self.point(p2), round(width * SCALE))
            else:
                pygame.draw.line(self.surface, color, self.point(start), self.point(end), round(width * SCALE))
        if svg:
            coords = ' '.join(f'{x},{y}' for x, y in points)
            dash = ' stroke-dasharray="10 7"' if dashed else ''
            self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                            f'stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        if arrow:
            x, y = points[-1]
            px, py = points[-2]
            length = math.hypot(x - px, y - py)
            ux, uy = (x - px) / length, (y - py) / length
            tip = [(x, y), (x - 11 * ux + 5 * uy, y - 11 * uy - 5 * ux),
                   (x - 11 * ux - 5 * uy, y - 11 * uy + 5 * ux)]
            pygame.draw.polygon(self.surface, color, [self.point(p) for p in tip])
            self.svg.append(f'<polygon points="{" ".join(f"{a},{b}" for a, b in tip)}" fill="{color}"/>')

    def circle(self, x, y, radius, fill='white', stroke=TEAL):
        pygame.draw.circle(self.surface, fill, self.point((x, y)), radius * SCALE)
        pygame.draw.circle(self.surface, stroke, self.point((x, y)), radius * SCALE, 2 * SCALE)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def box(self, x, y, w, h, title, lines=(), color=TEAL, fill=TEAL_BG, dashed=False, title_size=26, body_size=21):
        self.rect(x, y, w, h, fill, color, dashed)
        gap = 29
        total = 36 + gap * len(lines)
        top = y + (h - total) / 2
        self.text(x + w / 2, top + 18, title, title_size, color, True, 'middle', w - 24)
        for i, value in enumerate(lines):
            self.text(x + w / 2, top + 52 + i * gap, value, body_size, INK, anchor='middle', max_width=w - 24)

    def footer(self, value):
        self.path([(70, 843), (1530, 843)], '#D8E1E6', 1)
        self.text(70, 870, value, 18, MUTED, max_width=1460)

    def save(self, stem):
        png, svg = OUT / f'{stem}.png', OUT / f'{stem}.svg'
        pygame.image.save(self.surface, str(png))
        svg.write_text('\n'.join(self.svg + ['</svg>']), encoding='utf-8')
        print(png)
        print(svg)


def network():
    f = Figure('中国象棋 · 策略—价值网络架构', '深度学习部分｜默认 64 通道、6 个残差块｜尺寸省略批量维度 B')
    f.box(70, 178, 420, 120, '棋盘输入 · 15 × 10 × 9', ['红方 7 层 + 黑方 7 层', '当前走方 1 层'])
    f.box(570, 178, 420, 120, '初始卷积 · 3×3，15 → 64', ['BatchNorm + ReLU', '输出：64 × 10 × 9'])
    f.box(1070, 178, 460, 120, '残差主干 · ResidualBlock × 6', ['每块两次 3×3 卷积，含跳跃连接', '输出：64 × 10 × 9'], title_size=25)
    f.path([(490, 238), (570, 238)], TEAL, arrow=True)
    f.path([(990, 238), (1070, 238)], TEAL, arrow=True)
    f.path([(1300, 298), (1300, 345), (435, 345), (435, 397)], TEAL, arrow=True)
    f.path([(1175, 345), (1175, 397)], TEAL, arrow=True)
    f.text(780, 371, '共享棋盘特征 → 两个输出分支', 20, MUTED, anchor='middle')

    f.box(90, 397, 690, 98, '策略头 · 1×1 卷积，64 → 16', ['BatchNorm + ReLU → 16 × 10 × 9'], BLUE, BLUE_BG)
    f.box(90, 535, 690, 92, '展平 + 全连接', ['1440 → 8100'], BLUE, BLUE_BG)
    f.box(90, 667, 690, 92, '策略输出 · 8100 个动作分数（logits）', ['90 个起点 × 90 个终点'], BLUE, BLUE_BG)
    f.path([(435, 495), (435, 535)], BLUE, arrow=True)
    f.path([(435, 627), (435, 667)], BLUE, arrow=True)
    f.text(435, 799, '搜索阶段：过滤非法动作 + Softmax → 先验概率 p', 21, BLUE, anchor='middle', max_width=700)

    f.box(850, 397, 650, 98, '价值头 · 1×1 卷积，64 → 8', ['BatchNorm + ReLU → 8 × 10 × 9'], PURPLE, PURPLE_BG)
    f.box(850, 535, 650, 92, '展平 + 全连接', ['720 → 64，接 ReLU'], PURPLE, PURPLE_BG)
    f.box(850, 667, 650, 92, '全连接 64 → 1，接 Tanh', ['价值输出：v ∈ [−1, 1]'], PURPLE, PURPLE_BG)
    f.path([(1175, 495), (1175, 535)], PURPLE, arrow=True)
    f.path([(1175, 627), (1175, 667)], PURPLE, arrow=True)
    f.text(1175, 799, '当前落子方视角：预测最终胜负得分', 21, PURPLE, anchor='middle', max_width=650)
    f.footer('依据：net.py / encoder.py / mcts.py。策略先验 p 和局面价值 v 共同指导 MCTS 搜索。')
    f.save('01_网络架构')


def training():
    f = Figure('中国象棋 · 棋谱预训练与自对弈训练', '先通过棋谱学习落子，再通过搜索与自对弈迭代模型；图中明确区分实现状态')
    f.text(70, 166, '拟扩展 · 棋谱监督预训练（当前项目尚未实现）', 24, AMBER, True)
    f.box(80, 205, 420, 110, '高手棋谱', ['读取完整对局与最终胜负', '按整盘划分训练集 / 验证集'], AMBER, AMBER_BG, True)
    f.box(590, 205, 420, 110, '还原局面与训练标签', ['输入：每步落子前的局面', '标签：高手走法 a*、胜负 z'], AMBER, AMBER_BG, True)
    f.box(1100, 205, 420, 110, '监督预训练', ['策略头模仿高手走法', '价值头学习棋谱最终结果'], AMBER, AMBER_BG, True)
    f.path([(500, 260), (590, 260)], AMBER, dashed=True, arrow=True)
    f.path([(1010, 260), (1100, 260)], AMBER, dashed=True, arrow=True)
    f.path([(1310, 315), (1310, 365), (290, 365), (290, 460)], AMBER, dashed=True, arrow=True)
    f.text(800, 345, '以预训练权重作为自对弈起点', 20, AMBER, anchor='middle')
    f.text(590, 421, '已实现 · 搜索指导的自对弈强化学习', 24, TEAL, True)

    f.box(80, 460, 420, 116, '① 策略—价值网络', ['输出策略先验 p 与局面价值 v', '当前支持从随机权重开始'])
    f.box(590, 460, 420, 116, '② MCTS 搜索', ['多次模拟并更新访问次数', '得到搜索分布 π，选择落子'])
    f.box(1100, 460, 420, 116, '③ 自对弈', ['同一网络轮流执红、执黑', '逐步记录局面，直到结束 / 截断'])
    f.box(1100, 667, 420, 116, '④ 构造样本（s, π, z）', ['s：局面；π：归一化访问次数', 'z：从当前落子方视角标注结果'])
    f.box(590, 667, 420, 116, '⑤ 经验回放池', ['汇集不同对局的训练样本', '随机打乱，以小批量训练'])
    f.box(80, 667, 420, 116, '⑥ 联合优化网络', ['策略交叉熵 + 0.5 × 价值均方误差', '− 0.01 × 策略熵；Adam 更新'], body_size=20)
    f.path([(500, 518), (590, 518)], TEAL, arrow=True)
    f.path([(1010, 518), (1100, 518)], TEAL, arrow=True)
    f.path([(1310, 576), (1310, 667)], TEAL, arrow=True)
    f.path([(1100, 725), (1010, 725)], TEAL, arrow=True)
    f.path([(590, 725), (500, 725)], TEAL, arrow=True)
    f.path([(290, 667), (290, 576)], TEAL, arrow=True)
    f.text(320, 620, '更新权重', 21, TEAL)
    f.text(800, 620, '训练 → 新网络 → 新对局 → 新样本', 23, MUTED, anchor='middle')
    f.footer('z：获胜 +1、失败 −1、和棋 0；当前代码将达到最大步数的截断对局也记为 0。依据：train.py / ai.py / mcts.py。')
    f.save('02_训练流程与棋谱预训练')


def residual():
    f = Figure('中国象棋 · 残差块内部结构', '预激活残差块｜默认通道数 64｜主干连续堆叠 6 次')
    f.text(800, 198, '恒等分支：直接传递输入 x', 26, TEAL, True, 'middle')
    f.path([(200, 440), (200, 240), (1320, 240), (1320, 408)], TEAL, arrow=True)
    f.text(745, 330, '主分支 F(x)：两次归一化、激活与卷积', 24, MUTED, anchor='middle')
    f.box(50, 390, 120, 100, '输入 x', [], title_size=25)
    blocks = [(240, 120, 'BN', []), (395, 120, 'ReLU', []),
              (550, 150, '3×3 卷积', ['64 → 64']), (740, 120, 'BN', []),
              (895, 120, 'ReLU', []), (1050, 150, '3×3 卷积', ['64 → 64'])]
    for x, w, title, lines in blocks:
        f.box(x, 390, w, 100, title, lines, BLUE, BLUE_BG, title_size=24, body_size=20)
    for start, end in [(170, 240), (360, 395), (515, 550), (700, 740), (860, 895), (1015, 1050), (1200, 1288)]:
        f.path([(start, 440), (end, 440)], BLUE, arrow=True)
    f.circle(1320, 440, 32)
    f.text(1320, 437, '+', 40, TEAL, True, 'middle')
    f.path([(1352, 440), (1410, 440)], TEAL, arrow=True)
    f.box(1410, 390, 140, 100, '输出', [], title_size=25)
    f.text(1480, 528, 'x + F(x)', 27, TEAL, True, 'middle')
    f.text(800, 638, '输入 / 输出尺寸保持：64 × 10 × 9', 31, INK, True, 'middle')
    f.text(800, 695, '卷积核 3×3，步长 1，padding = 1；BN 和 ReLU 位于每次卷积之前', 24, MUTED, anchor='middle')
    f.text(800, 743, '跳跃连接为信息和梯度提供直接通路；本项目相加后直接返回', 24, MUTED, anchor='middle')
    f.footer('依据：net.py 中 ResidualBlock.forward()。该残差块不改变棋盘的空间尺寸。')
    f.save('03_残差块结构')


if __name__ == '__main__':
    pygame.font.init()
    OUT.mkdir(parents=True, exist_ok=True)
    network()
    training()
    residual()
    pygame.font.quit()
