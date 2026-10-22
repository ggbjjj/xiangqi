"""棋子类：颜色 + 类型。"""


class Piece:
    """一枚棋子。

    - color: 'r' 红方 或 'b' 黑方
    - kind:  'K','A','E','H','R','C','P'
    """

    __slots__ = ('color', 'kind')

    def __init__(self, color, kind):
        self.color = color
        self.kind = kind

    def __repr__(self):
        return f"{self.color}{self.kind}"

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.color == other.color and self.kind == other.kind

    def __hash__(self):
        return hash((self.color, self.kind))
