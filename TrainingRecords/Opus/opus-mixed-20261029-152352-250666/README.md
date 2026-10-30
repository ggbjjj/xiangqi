# Opus 追加三轮训练与参数导入

2026-10-30 从 `D:\模型参数信息\return-Opus-20261030-111815-696692.tar.gz` 导入。服务器本次从 ID2（累计完成 1 轮）追加 3 轮，未中断，累计完成 4 轮。训练 run_id 为 `opus-mixed-20261029-152352-250666`。

当前本地系列默认参数及游戏公开名称 **Opus-5** 均指向 **ID5 / Opus-5-e4-9cb66110**；本次运行的 `best` 与 `latest` 也都是 ID5。原 ID1–ID4 及第一轮日志保留。服务器原包的默认和公开名称仍绑定 ID2，本次只更新本地绑定。

| 本次轮次 | 参数 ID | 累计实际完成轮数 | 验证平均切片困惑度 | 验证困惑度中位数 |
|---|---|---:|---:|---:|
| 1 | ID3 | 2 | 9.517251 | 8.637302 |
| 2 | ID4 | 3 | 8.595137 | 7.666959 |
| 3 | ID5 | 4 | **8.000253** | **7.068653** |

ID5 固定检查点：`9cb661107ad3460eaa1c7f99572aa1a0`。权重 SHA-256：`d1ed7d9d8ec99728548561fe7d4603c948ed303d18266912524d2568aa79517a`。全网可训练参数 5,156,553 个，价值头已训练，指标 `value_trained=true`。

## 训练与验证配置

| 项目 | 配置或结果 |
|---|---|
| 数据来源 | `books`，`book_source=both`，主棋谱库与残局库合读去重 |
| 训练规模 | 29,181 原局 → 84,853 片段，T=32 |
| 验证规模 | 1,687 原局 → 4,830 切片，L=32，n=16 |
| 损失系数：策略 / 价值 / PPO / 动作熵 | 1 / 1 / 0 / 0 |
| 逻辑批量 / 微批量 / TBPTT | 128 / 64 / 16 |
| 学习率 / 权重衰减 / Dropout | 0.0003 / 0.0001 / 0.1 |
| 服务器设备 / 精度 | CUDA / `none`，未启用混合精度 |
| 结束时 global_step | 2,184；起始为 195，本次实际更新 1,989 次 |
| ID5 验证交叉熵 | 2.1814851564，有效步统计 |
| ID5 验证平均切片困惑度 | 8.0002534242，每片等权，用于选 best |
| ID5 语料几何平均困惑度 | 6.6183469823，额外诊断指标 |

填充步按正确动作概率 0.5 计入切片困惑度。三轮切片划分和评分方式一致。第一轮旧运行采用不同数据范围和评分方式，不能把其旧命中率或交叉熵直接当作本次平均切片困惑度比较。未执行新的完整验证集评估或棋力对战；上述验证结果来自服务器原始记录。

## 保存位置与校验

- 参数库：`ModelParameters/Opus/models.sqlite3`，包含 ID1–ID5 的权重、训练状态及每轮指标。
- 原始训练日志：`runs/mixed/opus-mixed-20261029-152352-250666/`，包括三轮逐片困惑度、PNG/SVG 区间图、TensorBoard、config、metrics、report、best 和 last_completed。
- 导入前备份：`backups/Opus/before-import-20261030-111815/models.sqlite3`，由 SQLite backup API 生成。
- [结构化训练与导入信息](training_summary.json)：来源、归档和数据库 SHA-256、旧默认绑定、校验结果及 CPU 加载检查。
- [源导出校验清单](source_manifest.json)：21 项文件大小与 SHA-256 全部通过校验。原始回传包和项目中的解包快照保持不变。

数据库完整性、外键检查通过。旧 ID1、ID2 的全部检查点字段与源库一致；安装后的全部五个版本权重和训练状态与源库逐字节一致。三份新权重均严格加载成功，ID5 的 AdamW 状态可恢复；CPU 前向输出有限，策略输出形状为 `[1,8100]`，价值输出为 `[1,3]`，初始局面的合法动作概率归一化检查通过。HaiKu 参数、旧迁移备份及两份数据集 SHA-256 均未改变。

本地仅更改默认指针和 Opus-5 公开名称绑定，因此安装后的数据库整体 SHA-256 与原始导出库不同；权重和训练状态的 SHA-256 保持一致。SQLite、runs、backups 和原始快照单独传输，不随 Git 上传。

## 查询、加载与下次训练

在项目根目录查看本次训练：

```bash
python -m training.runs --family Opus --run-id opus-mixed-20261029-152352-250666
python model_store.py --family Opus get-default
python model_store.py --family Opus get-latest
```

本地 CPU 环境启动游戏，开局时确认模型：

```powershell
uv run --locked --extra cpu xiangqi gui --family Opus --id ID5
```

服务器用当前共用训练入口继续追加 1 轮，沿用本次的主要配置：

```bash
uv run --locked --extra gpu python -m training --family Opus --source books --book-source both --epochs 1 --T 32 --L 32 --n 16 --batch-size 128 --micro-batch 64 --tbptt-steps 16 --lr 0.0003 --weight-decay 0.0001 --dropout 0.1 --lambda-policy 1 --lambda-value 1 --lambda-ppo 0 --lambda-entropy 0
```

此命令未指定 `--id` 和 `--version`，按服务器参数库中累计实际完成轮数最多的参数开始，轮数相同时选较大 ID。服务器若仍是本次导出时的库，将选择 ID5，从累计第 5 轮开始，完整结束后保存新 ID；`--epochs 1` 表示追加 1 轮。显式加 `--id ID5` 可固定起点。此入口加载权重后新建 AdamW，不恢复上次优化器；每轮保存的完整训练状态仍保留。旧 `resume_train.py` 是另一套完整优化器恢复入口，其 `--epochs` 为目标总轮数。

游戏和生成器省略 ID 时使用后台默认指针；训练自动选择最多轮数不会自动更新该指针或公开名称。若服务器也要将生成器默认设为 ID5，可在服务器执行 `python model_store.py --family Opus set-default --id ID5`；生成器显式指定 `--id ID5` 或 `--latest` 也可选择新参数。
