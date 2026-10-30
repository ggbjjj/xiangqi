# 中国象棋 · HaiKu / Opus

Pygame 桌面象棋，支持双人、人机（玩家任选红黑）、AI 观战。**HaiKu-5** 使用三个独立的历史/空间注意力核心；**Opus-5** 使用五个独立核心、双分支融合及策略/价值输出。也可选择内置搜索 AI 对弈。Opus 当前默认及游戏公开名称均使用 **ID5**，累计完成 4 轮训练，策略和价值头均已训练；结构和使用说明见 [Opus-5 模型与参数管理](docs/Opus-5模型与参数管理.md)。

## 启动游戏

Windows 双击 `启动象棋.bat`，或在项目目录运行：

```powershell
uv sync --locked --python 3.12 --extra cpu
uv run --locked --extra cpu xiangqi
```

主菜单数字键 1–4 选择模式；点棋子，再点绿色落点走子。U 悔棋，N 新局，B/Esc 返回菜单，空格暂停 AI 观战。

**人机对弈**：选择玩家执红/执黑 → 选择 `HaiKu-5`、`HaiKu-5.1`、`Opus-5` 或内置搜索 → 点击「开始对弈」。开局前必须确认模型，本局内不能更换；侧栏显示固定的对手名称。悔棋和 AI 提示不改变对手，点击「新局」或返回菜单后可重新选择。模型计算失败时停止自动走棋，点击「重试AI」仍使用本局原模型。

**AI 观战（模型自对弈演示）**：选模型 →「下一步」→ 选择「探索采样」或「贪心最优」→「开始观战」。模型默认探索采样，温度 **τ（Tao）=1.2**、随机探索比例 **ε（epsilon）=0.02**，可直接修改输入框。先对所有合法动作的 logits 除以 τ 做 Softmax，再混入 ε 比例的合法动作均匀分布后抽样；τ 必须是正有限数，ε 在 0–1 之间。τ=1、ε=0 仍按概率抽样；「贪心最优」才是始终取概率最高的合法动作。红黑双方使用同一设置，侧栏显示当前模式及参数；悔棋、重试保留设置，新局可重选。内置搜索按搜索结果选步，不使用模型探索参数。人机对手和 AI 提示仍使用原概率排序。

**AI 提示**：点击侧栏按钮或按 H → 先选模型名称（例如 `HaiKu-5`、`HaiKu-5.1`）→ 显示最高概率的 3 个合法走法及概率 → 点建议标出落点 → 点棋盘确认。双人模式服务当前走方，人机模式仅在玩家回合可用。概率是网络走法分布，不是胜率。玩家界面不展示训练轮数、指标、运行 ID 或参数 ID。

当前模型库保留 `HaiKu-5`（累计训练 1 轮）、`HaiKu-5.1`（累计完成 2 轮），`Opus-5` 绑定累计完成 4 轮的 ID5。各轮历史参数保留。后台将公开名称绑定到选定的训练参数 ID 后，名称才进入列表；新训练轮次不会自动新增条目或替换已绑定参数。`--smoke` 权重不能发布到游戏。训练指标与原始日志见 [训练归档与续训](TrainingRecords/README.md)。

```powershell
uv run --extra cpu xiangqi gui --mode pve_red
uv run --extra cpu xiangqi gui --mode pve_black --depth 3
uv run --extra cpu xiangqi gui --mode ai_vs_ai --model "Opus-5"
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

图形界面的 `--model` 仅预选模型；省略时预选参数库设置的默认公开版本（当前 HaiKu-5.1），仍须确认开局。即使带 `--mode pve_red`、`--mode pve_black` 或 `--mode ai_vs_ai` 直接启动，也会先显示模型选择页；AI 观战还需确认走子设置。后台修改名称绑定只影响之后新开的对局，已开局的参数保持不变。

规则：将死/困毙判负；同一走子循环连续重复、相同局面与走方第三次出现时，由当前走方变招，合法动作掩码排除继续该循环的历史走法，不因重复直接判和。中途变招后不累计零散重复，红黑双方适用同一规则；无合法变招时判当前方负。连续 120 步不吃子仍判和。这是项目自定义规则，不判定赛事长将长捉责任。导入保留原棋谱，训练读取时仅使用符合当前规则的前缀，无法沿用的对局结果不作为价值标签。

## 数据、模型和训练

显卡型号、CUDA 最低版本的含义、NVIDIA 驱动基线、FP16/BF16 选择与自检命令统一见 [GPU 与 CUDA 兼容说明](GPU与CUDA兼容说明.md)。当前锁定 `PyTorch 2.9.1+cu128`，使用 CUDA 12.8 运行库，无需另装系统 CUDA Toolkit。

AutoDL 的当前训练、生成、参数 ID 查询、默认 ID 设置和中断保存命令统一见 [指令.md](指令.md)。训练默认加载累计实际完成轮数最多的 ID，追加轮数按实际完成累计；生成器默认每 16 局提交一次。

GitHub 仅同步代码、依赖与说明，数据集、模型参数、自弈经验、训练日志和 exports 回传包单独传输。首次 clone 后需另外上传所需 SQLite 文件；训练后可用 tools/export_training.py 打包对应模型参数与本次日志。上传清单、AutoDL 拉取代码和结果取回步骤见 [GitHub与AutoDL同步](docs/GitHub与AutoDL同步.md)。

普通依赖和 NVIDIA CUDA 运行库（包括 cuBLAS、cuDNN、NCCL 等）默认使用项目内配置的阿里云 PyPI 镜像，并发下载为 2；GPU 版 PyTorch 单独使用南京大学 CUDA 12.8 镜像，CPU 版保留官方源。pyproject.toml 与 uv.lock 同步提交，服务器拉取后直接执行 `uv sync --locked --python 3.12 --extra gpu` 即可，无需每次设置镜像环境变量。

`python -m training.dashboard --logdir runs --port 6006` 启动中文实时曲线窗口：浏览器访问 `http://127.0.0.1:6006/`，查看损失、验证切片困惑度区间图和最佳参数 ID。默认每 16 步反传记录一个点，整个逻辑 Batch 结束才更新参数；平均切片困惑度最低者记为本次 best，各轮参数仍保留。验证独立切片 `--L 32`、上下文 `--n 16`（n≤L），填充步按正确动作概率 0.5 评分；终端分别显示验证原局数与切片后局数。已有参数可用 `python -m training.evaluate --family Opus --id 2` 单独评估和画图。

自对弈生成自动记录红胜、黑胜、和棋局数和占比，日志默认在 `runs/selfplay/`。同一窗口选择自弈运行即可查看胜负分布、累计变化及**实际结束步数区间图**，切换“已完成”或“已保存”同步切换范围；提前将死按真实步数统计，未完成局不计入，步数区间不打印到终端。GPU 推理微批量可用 `--micro-batch 32 --save-every 32` 显式指定，默认 16。若原服务只读取 `runs/mixed`，改为 `--logdir runs` 即可同时查看训练与生成，详见 [指令.md](指令.md)。

`python -m selfplay.runs --family Opus` 查询生成 run_id、实际保存局数及红胜/黑胜/和棋；加 `--run-id 实际run_id` 查看采样参数和生成配置。自弈/混合训练必须传 `--rollout-run 实际run_id`，仅加载该批棋局，按 C 切片；同一模型的多次生成互不合并。新运行的 run_id 与日志目录名一致，旧目录名可通过查询解析成数据库编号。加 `--delete --run-id 实际run_id` 可原子删除该批运行、棋局及逐步记录，`--dry-run` 可预览；历史日志保留，完整命令见 [指令.md](指令.md#查询生成器-run_id-与胜负信息)。

`python -m training.runs` 列出训练 run_id；加 `--family Opus --run-id 实际训练run_id` 查询配置、学习率、完成轮数、参数 ID、检查点和验证困惑度，`--json` 输出结构化数据。命令只读日志和参数库，不加载权重或占用 GPU；没有结束报告时明确显示“结束状态未知”。自定义位置用 `--logdir` 和 `--model-db` 指定，完整用法见 [指令.md](指令.md#查询训练-run_id-与对应训练信息)。

Opus 古谱训练已读取终局结果，默认同时训练策略与价值。按策略、价值、PPO、动作熵的顺序，古谱系数默认 1/1/0/0，自弈和混合默认 1/1/1/1；四项分别用 `--lambda-policy`、`--lambda-value`、`--lambda-ppo`、`--lambda-entropy` 指定。曲线显示总损失与加权分项，系数为零的分项隐藏。客户端右侧显示已训练价值模型对当前局面的红胜、和棋、黑胜概率；其他对手模式使用 Opus 默认参数独立分析。

共用的棋谱/PPO/价值/熵损失、指定模型版本的自弈生成器，以及按模型系列分库的自弈数据，见 [混合训练与自弈接口](docs/混合训练与自弈接口.md)。原棋谱训练入口也使用共用损失接口。

两种模型已在每核心 5 个 Block 的注意力激活输出、FFN 输出之后各加入一个残差 Dropout：每核心 10 层，HaiKu 共 30 层，Opus 共 50 层，默认 `p=0.1`。`--dropout` 可配置，省略沿用检查点；旧权重可严格加载。监督训练开启，验证/对弈及有效 PPO 逻辑批量关闭。更新后的两种完整网络图与逐层子架构图见 [网络图目录](docs/diagrams/README.md)。

棋谱监督训练支持三种读取方式：`--source books --book-source main` 只读取约 3 万局的主棋谱库，`--book-source endgames` 只读取独立残局库，`--book-source both` 两库合读并去重。三者共用 GameDataset / BookDataset 的分段读取接口，验证集也跟随该选择；终端明确标注方式及各库原局数、片段数。完整三条命令见 [指令.md](指令.md#三种棋谱加载命令)。

自弈训练读取时使用 `--C 128`（默认值）将完整对局连续分段，512 步会成为 4 个 128 步样本，尾段不足则补零且不参与损失。每段恢复真实起始棋盘、走方和完整规则历史，继承原局终局结果；模型缓存按片段重建。原自弈库保留完整对局。`--T` 继续控制古谱训练与验证，与 C 独立；完整例子见 [指令.md](指令.md)。

| 路径 | 内容 |
|---|---|
| Data/dataset.sqlite3 | 61,857 条原始记录；30,312 局规则校验后可训练样本，已合入 CBL |
| Data/endgames.sqlite3 | 独立残局库：2,246 局有效残局；与主库合读后两库共 32,546 局，含验证/测试 |
| models/HaiKu/、models/Opus/ | 两种模型的网络、配置、损失函数；models/common.py 保存共享注意力基础组件 |
| ModelParameters/HaiKu/models.sqlite3 | HaiKu 各版本权重、说明、优化器状态与默认版本 |
| ModelParameters/Opus/models.sqlite3 | Opus 各版本权重、说明、优化器状态与默认版本 |
| train.py | 棋谱监督训练、验证、保存、最终测试 |
| runs/ | 实时 TensorBoard、损失 CSV/PNG、最终评估 JSON |
| TrainingRecords/ | 已导入训练记录、校验清单、原始快照、下次续训配置 |

按完整棋局分组：27,179 训练 / 1,576 验证 / 1,557 测试。CBL 的 3,389 条记录已解码，3,362 条合格主线经去重新增 2,553 局；17 条空谱、10 条非法主线保留原文并隔离。保留真实起始局面和胜负；4 局来源胜负冲突已标记，停用其价值标签。步数统计、备份和来源见 [Data/README.md](Data/README.md)。重新导入：

```powershell
uv run --extra data xiangqi-import --workers 4
```

只合并现有库中待处理 CBL 并生成步数统计图：`uv run --extra cpu python -m data_pipeline.merge_cbl`。命令会先备份数据库；已按当前解码版本处理的记录会跳过。

HaiKu 默认输入 [T,B,15,10,9]，输出 [T,B,8100]；`--T` 是古谱读取时的片段长度，默认 120，按红黑合计单步计数。完整棋局按 T 连续分段：100 步、T=32 得到 4 个样本，有效长度为 32、32、32、4，尾段补零；每段恢复真实起始棋盘和完整规则历史，模型缓存从该段起点建立。古谱训练、验证、测试均使用同一读取器，各段保持原局的数据划分和胜负。终端显示分段后的训练总数及原始局数；batch=128 和 micro-batch 均按片段计数，损失仅按有效步计算。

`--context-plies` 独立控制 HaiKu 九对 K/V 缓存的滑动窗口（局面数，包含当前局面），新模型及古谱训练默认 **30**，不随 T 改变。例如 `--T 1000 --context-plies 30` 每段最多 1000 步，每对缓存仅保留该段内最近 30 个局面，片段内时间索引继续递增。共用训练入口每 16 步截断梯度，旧 `train.py` 的 HaiKu 按片段完整反传；两者每个新微批量均重建空缓存、时间从 0 开始，梯度累积不携带上一微批的 K/V。同一微批量内部跨反传段保留缓存。Opus 保持原来的五个独立窗口。

常用 HaiKu 古谱入口加载旧权重后也默认改为 30，并保存到新 ID；旧参数记录保持原窗口。独立评估、游戏、生成器继续使用所选检查点的窗口，自弈/混合训练默认跟随采样版本；PPO 不接受窗口与采样不匹配。网络图已同步更新，详见 [网络图目录](docs/diagrams/README.md)。

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

优化器续训按记录或检查点保存的窗口；当前续训记录已设为 30。可独立覆盖，例如 `python resume_train.py --epochs 3 --context-plies 30`，或在续训配置中添加 `"context_plies": 30`。窗口变化时重新比较本次窗口下的 best，旧检查点不改写。新窗口随新检查点保存，游戏推理自动使用该值。`--T`、数据划分和 seed 仍按原续训校验保持一致。

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
