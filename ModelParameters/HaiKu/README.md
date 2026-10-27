# HaiKu 参数

运行时文件为 `models.sqlite3`，包含本系列全部参数 ID、版本说明、默认指针、权重、优化器与指标。数据库不随 Git 上传，首次部署须单独传输已有库；后续拉取代码保留服务器现有库。

训练后取回此数据库和对应的 `runs/` 运行目录。可用 `python tools/export_training.py --family HaiKu --run-dir runs/mixed/实际run_id` 导出一致快照。
