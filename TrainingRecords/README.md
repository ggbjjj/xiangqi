# 训练记录与下次续训

当前已累计完成 **2 轮**：第二轮从第一轮检查点继续训练，完成训练、验证和最终测试集评估。两轮参数及全部历史归档保留。

| 游戏模型名称 | 已完成轮数 | 参数 ID | 训练记录 |
|---|---:|---|---|
| HaiKu-5 | 1 | ddba0b0639364a4cbd663f51ab74d3d7 | [第一轮归档](haiku5-20261025-105746-759121/训练记录与续训说明.md) |
| HaiKu-5.1 | 2 | e2bbff5da4cb42a29484fd1d03a095cb | [第二轮归档](haiku5-20261025-105746-759121-epoch2-20261026/训练记录与续训说明.md) |

`ModelParameters/HaiKu/models.sqlite3` 同时存储两轮权重和优化器状态；`HaiKu-5` 保持第一轮参数，`HaiKu-5.1` 使用第二轮并设为默认。根目录旧参数库保留为迁移备份；Opus 参数独立存放在 `ModelParameters/Opus/models.sqlite3`。玩家列表仅显示模型名称，训练轮次、指标和参数 ID 只在后台记录。

`current_resume.json` 现指向第二轮检查点，默认再完成第 3 轮。第一轮固定恢复配置仍在 `haiku5-20261025-105746-759121/resume_config.json`，导入第二轮前的当前配置也保存在新归档 `previous_current_resume.json`。

## 在 AutoDL 继续第三轮

在项目根目录执行，保持 GPU 环境：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 只核验参数、优化器及数据划分，不开始训练
uv run --locked --extra gpu python resume_train.py --check-only

# 从已完成第2轮继续第3轮，达到总计3轮
uv run --locked --extra gpu python resume_train.py --epochs 3
```

不指定 `--epochs` 时当前默认也是 3。若目标是总计 10 轮，改为 `--epochs 10`，将继续第 3–10 轮。`--epochs` 表示目标总轮数，不是额外轮数。逻辑 batch=128、micro-batch=16、T=120、seed=42，保持已验证配置。

K/V 滑动窗口现可独立设置：`python resume_train.py --epochs 3 --context-plies 64`。未指定时优先使用记录中的可选 `context_plies` 字段，再沿用检查点窗口；现有记录不需要修改。窗口随新检查点保存，旧参数和记录不改写；窗口变化时重新选择该窗口下的 best。T、数据划分和 seed 的原校验不变。

当前入口恢复第二轮权重、AdamW 和随机数状态，每次为新日志建立独立目录 `runs/resumed/时间目录/原run_id/`。历史 CSV 接续已有两轮指标，旧日志和旧参数不改写。最终测试不用于选择模型，best 按验证 loss 选取。

AutoDL 上如果仍是第一轮的续训配置，上传当前指向第二轮的 `TrainingRecords/current_resume.json` 后再运行。同步更新 `train.py`、`resume_train.py`、`model_store.py`、`project_paths.py` 和整个 `models/`；现有检查点和续训记录兼容。服务器若仍使用根目录旧库，先运行 `python model_store.py migrate`，再使用当前新路径续训。服务器已有两轮模型参数时保留该参数库，无需用本地库覆盖。

如需明确从第一轮另起一条训练分支，可指定历史记录：

```bash
uv run --locked --extra gpu python resume_train.py --record TrainingRecords/haiku5-20261025-105746-759121/resume_config.json --epochs 2
```

## 文件与保留策略

- 第二轮日志：`runs/resumed/20261026-003904-059919/haiku5-20261025-105746-759121/`，包含原始 config、两轮 history.csv/history.png、resume_origin、TensorBoard 和 evaluation.json。
- 第二轮完整原始快照、结构化训练信息及校验清单：`haiku5-20261025-105746-759121-epoch2-20261026/`。
- 第一轮归档与原始日志保持原样，参数和训练状态逐字节不变。
- 新下载的第一轮 TensorBoard 文件比旧文件多 110 字节。项目保留原旧文件，新文件完整存放在第二轮归档的 `source/` 中，两份均保留，不直接覆盖同名旧日志。
- 导入前的项目参数库已另行备份，位置写入第二轮 `training_summary.json`。

`source/` 原始快照、SQLite 参数库和 runs 均在磁盘保留，不随 Git 提交或构建包自动携带。迁移新实例时须单独携带数据集、已训练参数库、续训配置和所需历史日志；不要用旧空参数库覆盖已训练数据。
