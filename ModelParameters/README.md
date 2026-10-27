# 模型参数库

`HaiKu/models.sqlite3` 和 `Opus/models.sqlite3` 分别存储对应系列的架构配置、各版本权重 BLOB、版本名称与说明、默认版本、优化器状态、训练进度、评估指标及 SHA-256。网络代码统一放在项目的 `models/HaiKu/` 与 `models/Opus/`。

HaiKu 当前默认版本为 **HaiKu-5.1**；Opus 当前默认版本为 **Opus-5.initial**（未训练，不进入玩家对手列表）。后台 `load_default_model('HaiKu')` / `load_default_model('Opus')` 自动加载各自默认参数。完整目录、命令和 Opus 训练说明见 [Opus-5 模型与参数管理](../docs/Opus-5模型与参数管理.md)。

根目录旧的 `models.sqlite3` 保留为原始备份，迁移后不再使用它写入；旧 HaiKu 续训路径由后台兼容转向新库。迁移校验记录在 `migration_20261027.json`，两轮权重、优化器和公开名称绑定均保留。

当前库已从 `D:\模型参数信息` 导入两轮正式训练权重，第一轮和第二轮分别保存，不覆盖或删除。原始日志、指标与恢复配置见 [训练归档](../TrainingRecords/README.md)。正常训练每个 epoch 提交一个事务保存检查点，`latest` 和 `best` 别名按 run_id 分组。

| 玩家可见名称 | 累计已完成轮数（仅后台记录） | 参数 ID |
|---|---:|---|
| HaiKu-5 | 1 | ddba0b0639364a4cbd663f51ab74d3d7 |
| HaiKu-5.1 | 2 | e2bbff5da4cb42a29484fd1d03a095cb |

两份参数属于同一网络、同一次训练的连续轮次。`HaiKu-5` 的绑定保持第一轮；`latest` 和 `best` 都指向第二轮。当前续训配置已指向第二轮，下次从第 3 轮开始。第一轮权重和训练状态已逐字节核验未改写。
`checkpoints` 是后台训练记录；每轮有独立的 `checkpoint_id`（参数 ID）。`model_releases` 存储公开名称与参数 ID 的绑定。当前 `HaiKu-5` 对应第 1 轮，`HaiKu-5.1` 对应第 2 轮；名称由管理员决定，不由训练轮数自动生成。游戏列表仅返回 `{"name":"HaiKu-5"}` 这样的名称，不返回训练信息或参数 ID。

只有 `published=1` 且 `global_step>0` 的检查点允许绑定；这里 `published` 是检查点的可发布标志，不等于自动加入玩家列表。`--smoke` 检查点不可绑定。训练保存新轮次不改变既有名称对应的参数。
KV 缓存属于当前棋局，不保存进数据库。换模型、悔棋或换局时重建。

从服务器取回参数：训练保存完成后复制该 SQLite 文件到本目录；首次复制可替换初始空库，已有模型时先备份，或用 SQLite backup API 取得一致快照后另存，再通过 `--model-db` 指定。
不要复制 `.venv`；CPU 和 CUDA PyTorch 可以加载相同 state_dict。

## 后台查看与启动绑定

在项目根目录执行，查看全部训练参数的 ID、run_id、epoch、global_step、指标和保存时间。`can_publish=false` 表示测试或未训练检查点，仅供后台检查，不能供玩家使用：

```powershell
uv run --locked --extra cpu python model_store.py list-checkpoints
```

首次绑定、添加名称或更换同名模型的参数，都可以在启动时直接指定：

```powershell
uv run --locked --extra cpu xiangqi gui --model "HaiKu-5" --checkpoint 实际参数ID
uv run --locked --extra cpu xiangqi gui --model "HaiKu-5.1" --checkpoint 另一个参数ID
```

请替换上述参数 ID；不要输入占位文字。该配置写入 `model_releases` 并跨启动保留，同名更新不会重复增加列表条目。只有 `--checkpoint`、没有 `--model` 会报错，避免不明确的绑定。

以后启动无需重复参数 ID；图形界面的 `--model` 预选人机/观战模型，省略时预选默认公开版本，开局前仍须在客户端确认。AI 提示仍由玩家在全部已发布名称中单独选择，不更换本局对手：

```powershell
uv run --locked --extra cpu xiangqi gui --model "HaiKu-5"
```

进入对局后参数固定：同名模型在后台重新绑定、悔棋、选择另一个提示模型均不改变当前对手；模型出错只允许重试本局参数或退出，不自动切换模型。点击新局会先重新选择。CLI 对弈可用 `--model 名称` 固定对手，省略时先加载默认公开版本；没有配置默认时才使用列表首个已发布名称。

也可以只管理绑定，不启动游戏；`--db` 放在子命令前面，省略则使用本目录的默认库：

```powershell
uv run --locked --extra cpu python model_store.py --family HaiKu list-releases
uv run --locked --extra cpu python model_store.py --family HaiKu publish --name "HaiKu-5" --checkpoint 实际参数ID
uv run --locked --extra cpu python model_store.py --family HaiKu publish --name "HaiKu-5.1" --checkpoint 另一个参数ID
uv run --locked --extra cpu python model_store.py --family HaiKu publish --name "HaiKu-5" --run-id 实际运行ID --alias best
uv run --locked --extra cpu python model_store.py --family HaiKu unpublish --name "HaiKu-5.1"
```

`--alias best` 在发布时解析成固定参数 ID；后续训练产生新的 best 不会自动替换该名称。`--order 0` 可指定排序值，数值小的在前。下架仅隐藏名称，保留所有训练参数；重新 publish 可恢复。

远程执行后台命令时将 `--extra cpu` 换为 `--extra gpu`，避免 uv 切换服务器的 PyTorch 环境。已有旧参数库在发布时自动增加映射表，保留原有参数、别名和训练记录，无需重新训练。**不要把本地初始空库上传覆盖服务器已训练的 models.sqlite3。**
