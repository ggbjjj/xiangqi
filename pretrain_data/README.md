# 原始棋谱下载目录

此目录的原始下载、压缩包和调查中间文件保留在本地，不随 Git 提交。使用已经整理好的 `Data/dataset.sqlite3` 训练时，不需要上传此目录到 AutoDL。

需要重新导入原始棋谱时，再单独传输 `source_survey_20261024/`，然后运行 `python -m data_pipeline.import_records --workers 4`。来源、数据处理与授权信息见 [Data/README.md](../Data/README.md)。

残局原始下载及来源核验位于本地 `endgame_survey_20261029/`，包含 14 套 CBL、校验清单和来源报告。`python -m data_pipeline.import_endgames` 从这些文件及原棋谱库中的 CCPD 残局原文构建独立 `Data/endgames.sqlite3`。服务器使用已建好的残局库或运行 `--report-only` 统计时，无需上传这些原始文件。
