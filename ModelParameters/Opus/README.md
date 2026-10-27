# Opus 参数

运行时文件为 `models.sqlite3`，包含本系列全部参数 ID、版本说明、默认指针、权重、优化器与指标。数据库不随 Git 上传，首次部署须单独传输已有库；后续拉取代码保留服务器现有库。

确实需要从零初始化时才运行 `python model_store.py --family Opus initialize`。继续已有训练应上传原有数据库。训练后可用 `python tools/export_training.py --family Opus --run-dir runs/mixed/实际run_id` 导出。
