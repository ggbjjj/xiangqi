# HaiKu 自弈数据

`games.sqlite3` 独立保存所有 HaiKu 参数版本生成的棋局，使用 runs 表记录确切版本、参数 ID 和权重 SHA-256，不按版本拆库。数据库首次生成时自动创建。

调用示例：`python -m selfplay --family HaiKu --version "HaiKu-5.1" --games 10 --device cpu`。当前 HaiKu 无价值头，使用明确标记的零基线；训练时价值权重设为 0。

详见 `docs/混合训练与自弈接口.md`。
