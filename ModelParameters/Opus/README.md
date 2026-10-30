# Opus 参数

运行时文件为 `models.sqlite3`，包含本系列全部参数 ID、版本说明、默认指针、权重、优化器与指标。数据库不随 Git 上传，首次部署须单独传输已有库；后续拉取代码保留服务器现有库。

当前默认为 **ID5：Opus-5-e4-9cb66110**，公开名称 **Opus-5** 也绑定 ID5。固定检查点为 `9cb661107ad3460eaa1c7f99572aa1a0`。本次从 ID2 追加训练 3 轮，累计实际完成 **4 轮**，ID1–ID4 全部保留。追加训练的策略/价值/PPO/动作熵系数为 `1/1/0/0`，价值头已训练；验证平均切片困惑度为 **8.000253**（L=32，n=16；1,687 原局 → 4,830 切片）。[本次训练记录与导入校验](../../TrainingRecords/Opus/opus-mixed-20261029-152352-250666/README.md)。

`python -m training --family Opus` 省略 `--id` 和 `--version` 时自动加载累计实际完成轮数最多的参数，轮数相同时选较大的 ID；当前选择 ID5。游戏和生成器省略 ID 时使用后台默认 ID。两种选择规则独立，训练保存新版本不会自动更换游戏公开名称的绑定。

确实需要从零初始化时才运行 `python model_store.py --family Opus initialize`。继续已有训练应上传原有数据库。训练后可用 `python tools/export_training.py --family Opus --run-dir runs/mixed/实际run_id` 导出。
