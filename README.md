# 中国象棋 · HaiKu / Opus

Pygame 桌面象棋，支持双人、人机（玩家任选红黑）、AI 观战。**HaiKu-5** 使用三个独立的历史/空间注意力核心；**Opus-5** 使用五个独立核心、双分支融合及策略/价值输出。也可选择内置搜索 AI 对弈。Opus 当前参数为未训练初始化，未发布为游戏对手；结构和使用说明见 [Opus-5 模型与参数管理](docs/Opus-5模型与参数管理.md)。

## 启动游戏

Windows 双击 `启动象棋.bat`，或在项目目录运行：

```powershell
uv sync --locked --python 3.12 --extra cpu
uv run --locked --extra cpu xiangqi
```

主菜单数字键 1–4 选择模式；点棋子，再点绿色落点走子。U 悔棋，N 新局，B/Esc 返回菜单，空格暂停 AI 观战。

**人机对弈**：选择玩家执红/执黑 → 选择 `HaiKu-5`、`HaiKu-5.1` 或内置搜索 → 点击「开始对弈」。开局前必须确认模型，本局内不能更换；侧栏显示固定的对手名称。悔棋和 AI 提示不改变对手，点击「新局」或返回菜单后可重新选择。模型计算失败时停止自动走棋，点击「重试AI」仍使用本局原模型。AI 观战同样在开局前选择模型。

**AI 提示**：点击侧栏按钮或按 H → 先选模型名称（例如 `HaiKu-5`、`HaiKu-5.1`）→ 显示最高概率的 3 个合法走法及概率 → 点建议标出落点 → 点棋盘确认。双人模式服务当前走方，人机模式仅在玩家回合可用。概率是网络走法分布，不是胜率。玩家界面不展示训练轮数、指标、运行 ID 或参数 ID。

当前模型库保留 `HaiKu-5`（累计训练 1 轮），新增 `HaiKu-5.1`（接着第一轮训练，累计完成 2 轮）。两个名称分别绑定各自参数，第一轮没有覆盖或删除。后台将公开名称绑定到选定的训练参数 ID 后，名称才进入列表；新训练轮次不会自动新增条目或替换已绑定参数。`--smoke` 权重不能发布到游戏。训练指标与原始日志见 [训练归档与续训](TrainingRecords/README.md)。

```powershell
uv run --extra cpu xiangqi gui --mode pve_red
uv run --extra cpu xiangqi gui --mode pve_black --depth 3
uv run --extra cpu xiangqi gui --model-db ModelParameters/HaiKu/models.sqlite3 --model "HaiKu-5"
uv run xiangqi cli pvp
```

纯规则游戏不需要 PyTorch，可直接 `uv run xiangqi`；本地训练或提示使用 `--extra cpu`，远程 CUDA 训练使用 `--extra gpu`。原 `--extra neural` 保留为 CPU 环境的兼容别名。CLI 坐标从 0 开始，例如 `60-50`；支持 u 悔棋、q 退出。

后台查看训练参数 ID，再在启动时指定名称对应的参数（把 `实际参数ID` 替换为列表中的 `checkpoint_id`）：

```powershell
uv run --locked --extra cpu python model_store.py list-checkpoints
uv run --locked --extra cpu xiangqi gui --model "HaiKu-5" --checkpoint 实际参数ID
```

绑定保存在 SQLite，下次只需 `--model "HaiKu-5"`；想换参数时，对同一名称重新传入另一个 ID。添加 `HaiKu-5.1` 可绑定另一轮参数，两个名称并存。后台发布、下架及指定参数库见 [ModelParameters/README.md](ModelParameters/README.md)。

图形界面的 `--model` 仅预选模型；省略时预选参数库设置的默认公开版本（当前 HaiKu-5.1），仍须点击「开始对弈」。即使带 `--mode pve_red` 或 `--mode pve_black` 直接启动，也会先显示模型选择页。后台修改名称绑定只影响之后新开的对局，已开局的参数保持不变。

规则：将死/困毙判负；桌面游戏使用三次重复或连续 120 步不吃子判和的休闲规则，未实现赛事长将长捉裁决。训练导入校验采用棋子合法性与将帅安全，保留真实棋谱完整历史。

## 数据、模型和训练

AutoDL 的当前训练、生成、参数 ID 查询、默认 ID 设置和中断保存命令统一见 [指令.md](指令.md)。训练默认加载累计实际完成轮数最多的 ID，追加轮数按实际完成累计；生成器默认每 16 局提交一次。

GitHub 仅同步代码、依赖与说明，数据集、模型参数、自弈经验、训练日志和 exports 回传包单独传输。首次 clone 后需另外上传所需 SQLite 文件；训练后可用 tools/export_training.py 打包对应模型参数与本次日志。上传清单、AutoDL 拉取代码和结果取回步骤见 [GitHub与AutoDL同步](docs/GitHub与AutoDL同步.md)。

`python -m training.dashboard --logdir runs --port 6006` 启动中文实时曲线窗口：浏览器访问 `http://127.0.0.1:6006/`，查看损失、独立古谱验证指标和最佳参数 ID。默认每 16 步反传记录一个点，整个逻辑 Batch 结束才更新参数；古谱验证损失最低者记为本次 best，各轮参数仍保留。

共用的棋谱/PPO/价值/熵损失、指定模型版本的自弈生成器，以及按模型系列分库的自弈数据，见 [混合训练与自弈接口](docs/混合训练与自弈接口.md)。原棋谱训练入口也使用共用损失接口。

| 路径 | 内容 |
|---|---|
| Data/dataset.sqlite3 | 61,857 条原始记录；27,759 局规则校验后可训练样本 |
| models/HaiKu/、models/Opus/ | 两种模型的网络、配置、损失函数；models/common.py 保存共享注意力基础组件 |
| ModelParameters/HaiKu/models.sqlite3 | HaiKu 各版本权重、说明、优化器状态与默认版本 |
| ModelParameters/Opus/models.sqlite3 | Opus 各版本权重、说明、优化器状态与默认版本 |
| train.py | 棋谱监督训练、验证、保存、最终测试 |
| runs/ | 实时 TensorBoard、损失 CSV/PNG、最终评估 JSON |
| TrainingRecords/ | 已导入训练记录、校验清单、原始快照、下次续训配置 |

按完整棋局分组：24,916 训练 / 1,440 验证 / 1,403 测试。CBL 古谱与赛事原文已入库，其中 3,389 条待主线解码，不混入训练集。数据口径与来源见 [Data/README.md](Data/README.md)。重新导入：

```powershell
uv run --extra data xiangqi-import --workers 4
```

HaiKu 默认输入 [T,B,15,10,9]，输出 [T,B,8100]；`--T` 指每局读取的最大单步总数，默认 120，红黑合计。`--context-plies` 独立控制 HaiKu 九对 K/V 缓存的滑动窗口（局面数，包含当前局面），新训练默认 120，不随 T 改变。例如 `--T 1000 --context-plies 120` 读取最多 1000 步，但每对缓存仅保留最近 120 个局面，时间索引继续递增。训练按有效步平均交叉熵，非法动作屏蔽，短局补零，长局截取连续前 T 步。batch=128 是逻辑批量，通过 micro-batch 梯度累积控制显存。各 batch 的 KV 缓存独立。HaiKu 保留完整有效序列反传；Opus 使用独立的五个窗口和默认 16 步梯度截断，详见上面的 Opus 说明。

选定 **Python 3.12 / PyTorch 2.9.1 / 远程 CUDA 12.8**；本地使用 CPU wheel。完整服务器安装、GPU 型号、显存调节、PyCharm 和 TensorBoard 配置见 [Haiku-5 训练与部署](docs/Haiku-5训练与部署.md)。

在项目根目录二选一执行，uv 自动创建 Python 环境并安装运行与训练依赖：

```bash
# 本地 CPU：PyTorch 2.9.1+cpu
uv sync --locked --python 3.12 --extra cpu

# AutoDL / NVIDIA GPU：PyTorch 2.9.1+cu128
uv sync --locked --python 3.12 --extra gpu
```

重新上传本次更新的 `pyproject.toml`、`uv.lock` 和 `.python-version`，不要上传本地 `.venv`。两种选项互斥，选择一种即可，不使用 `--all-extras`。GPU wheel 包含 CUDA 12.8 运行库，服务器须有兼容的 NVIDIA 驱动。

PyCharm 本地解释器选择 `D:\象棋\.venv\Scripts\python.exe`；远程 SSH 解释器选择 `远程项目目录/.venv/bin/python`，例如 `/root/autodl-tmp/xiangqi/.venv/bin/python`。源码目录映射仍为本地 `D:\象棋` → 远程项目目录。GPU 自检：`uv run --locked --extra gpu python tools/check_training_env.py`。

本机流程自检（不会发布模型）：

```powershell
uv run --extra cpu xiangqi-train --epochs 1 --T 3 --batch-size 2 --micro-batch 1 --limit-games 2 --device cpu --smoke --model-db artifacts/smoke.sqlite3 --output artifacts/smoke-runs
```

**继续当前已保存的第二轮**，在 AutoDL 项目根目录使用下面的专用入口；`--epochs 3` 只继续第 3 轮，不会从随机参数重训。该入口保留原日志，并给新日志建立独立目录：

```bash
uv run --locked --extra gpu python resume_train.py --check-only
uv run --locked --extra gpu python resume_train.py --epochs 3
```

目标总计 10 轮时改为 `--epochs 10`。具体检查点 ID、数据指纹与配置已保存在 `TrainingRecords/current_resume.json`；第一轮的恢复配置仍保存在其历史归档中。

续训不指定窗口时沿用检查点保存的值；可独立覆盖，例如 `python resume_train.py --epochs 3 --context-plies 64`，或在续训配置中添加 `"context_plies": 64`。窗口变化时重新比较本次窗口下的 best，旧检查点不改写。新窗口随新检查点保存，游戏推理自动使用该值。`--T`、数据划分和 seed 仍按原续训校验保持一致。

远程安装 GPU 环境后，在项目根目录的两个终端分别执行：

```bash
uv run --locked --extra gpu xiangqi-train --epochs 10 --T 120 --batch-size 128 --micro-batch 1 --workers 4 --device cuda --precision auto
uv run --locked --extra gpu tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

控制台实时进度和 loss；TensorBoard 实时记录每个微批量 loss，以及每轮训练/验证曲线。每个 epoch 保存 SQLite，验证集选择 best，训练结束用 best 评估测试集 Top-1/Top-3 和交叉熵。指标是棋谱动作预测准确度，不等于对弈胜率。取回服务器参数库后，在后台绑定公开名称即可供玩家选择。

## 验证与构建

```powershell
uv run --extra cpu pytest -q
uv build
```

本次源码备份在 `backups/haiku5_before/`；较早备份仍保留。大体积棋谱库、模型库与训练日志留在磁盘，已加入 Git 忽略规则；部署时须单独复制 Data 和 ModelParameters。
