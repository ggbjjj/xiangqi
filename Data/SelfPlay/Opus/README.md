# Opus 自弈数据

`games.sqlite3` 独立保存所有 Opus 参数版本生成的棋局，使用 runs 表记录确切版本、参数 ID 和权重 SHA-256，不按版本拆库。数据库首次生成时自动创建。

调用示例：`python -m selfplay --family Opus --version "Opus-5.initial" --games 10 --device cpu`。当前 Opus-5.initial 仍是未训练参数，价值预测不是监督标签；训练标签来自本次真实对局结果。

详见 `docs/混合训练与自弈接口.md`。
