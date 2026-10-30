# 自弈运行数据

HaiKu 与 Opus 分别使用 `HaiKu/games.sqlite3`、`Opus/games.sqlite3`，同系列不同参数版本共用该库。数据库及其临时侧文件不随 Git 提交。

新服务器不必携带空库，生成器会自动创建。继续使用既有自弈数据训练时需单独上传对应库；取回新生成的数据时可在导出命令添加 `--include-selfplay`。详见 [指令.md](../../指令.md)。

同一模型每次生成都有独立 run_id。自弈和混合训练必须指定 `--rollout-run`，读取函数只加载此 run 已保存的棋局，再按 C 切片；128 局和 1000 局两个 run 分开读取。`python -m selfplay.runs --family Opus` 列出运行及实际红胜、黑胜、和棋局数；加 `--run-id` 查询生成配置，加 `--delete` 原子删除这一批的运行、棋局及逐步记录，`--dry-run` 可预览。历史日志保留供追溯。

GPU 推理微批量用 `--micro-batch` 指定，例如 `--micro-batch 32 --save-every 32`；默认 16。共用窗口 `python -m training.dashboard --logdir runs --port 6006` 实时展示胜负及实际结束步数分布，步数区间只在窗口显示，提前终局按真实长度统计。
