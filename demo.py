"""快速演示：打印初始棋盘 + 走几步。"""

from game import Game

g = Game()
print("=== 初始棋盘 ===")
print(g.board)
print(f"turn = {g.turn}, status = {g.status!r}")
print(f"红方合法走法数: {len(g.all_legal_moves())}")

# 走几步：兵五进一, 炮二平五
print("\n--- 红方走 兵五进一 (6,2) -> (5,2) ---")
print(g.move(6, 2, 5, 2))
print(g.board)
print(f"turn = {g.turn}")

print("\n--- 黑方走 卒5进1 (3,2) -> (4,2) ---")
print(g.move(3, 2, 4, 2))
print(g.board)
print(f"turn = {g.turn}")

print("\n--- 红方走 炮二平五 (7,1) -> (7,4) ---")
# 炮从 (7,1) 到 (7,4)，中间隔 (6,1) 空、(5,1) 空、(4,1) 空，第4格(7,4)是空格 -> 不能直接吃
# 炮不吃子时可以滑动，所以可以直接走到 (7,4)
print(g.move(7, 1, 7, 4))
print(g.board)
