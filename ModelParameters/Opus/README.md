# Opus 参数

运行时文件为 `models.sqlite3`，包含本系列全部参数 ID、版本说明、默认指针、权重、优化器与指标。数据库不随 Git 上传，首次部署须单独传输已有库；后续拉取代码保留服务器现有库。

当前默认为 **ID2：Opus-5-e1-bcdb8df2**，已完整完成 **1 轮古谱策略训练**，公开名称为 **Opus-5**。固定检查点为 `bcdb8df21f5048a88c28f34efb949b8f`，初始化 ID1 保留。策略/价值/PPO/动作熵系数为 `1/0/0/0`，价值头尚未训练，不用于客户端局势预测。验证交叉熵 2.595229，首选/前三动作命中率 27.84% / 48.30%。[本次训练记录与曲线](../../TrainingRecords/Opus/opus-mixed-20261028-094701-460939/README.md)。

确实需要从零初始化时才运行 `python model_store.py --family Opus initialize`。继续已有训练应上传原有数据库。训练后可用 `python tools/export_training.py --family Opus --run-dir runs/mixed/实际run_id` 导出。
