# 原始棋谱下载目录

此目录的原始下载、压缩包和调查中间文件保留在本地，不随 Git 提交。使用已经整理好的 `Data/dataset.sqlite3` 训练时，不需要上传此目录到 AutoDL。

需要重新导入原始棋谱时，再单独传输 `source_survey_20261024/`，然后运行 `python -m data_pipeline.import_records --workers 4`。来源、数据处理与授权信息见 [Data/README.md](../Data/README.md)。
