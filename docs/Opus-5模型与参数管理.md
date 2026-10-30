# Opus-5 模型与参数管理

## 目录

```text
models/
  common.py              # 历史/空间/交叉注意力、序列缓存
  HaiKu/                 # 原 HaiKu 网络、配置与策略损失
  Opus/                  # 新 Opus 网络、配置与价值损失
  haiku5/                # 旧导入路径兼容入口
ModelParameters/
  HaiKu/models.sqlite3   # HaiKu 各参数版本
  Opus/models.sqlite3    # Opus 各参数版本
  models.sqlite3         # 迁移前原库备份
```

数据库保存权重和优化器 BLOB、架构配置、版本名称 `version_name`、说明 `description`、训练指标与 SHA-256。`model_defaults` 为每个系列指定一个默认检查点；`model_releases` 将游戏公开名称固定绑定到某个检查点，两个设置分别管理。

当前 HaiKu 默认是已完成两轮训练的 HaiKu-5.1，第一轮 HaiKu-5 仍保留。Opus 默认是 **ID5 / `Opus-5-e4-9cb66110`**，公开名称 Opus-5 也绑定 ID5，ID1–ID4 保留。本次从 ID2 追加 3 轮，累计实际完成 4 轮；追加训练的策略/价值/PPO/动作熵系数为 1/1/0/0，价值头已训练。验证平均切片困惑度为 8.000253（L=32，n=16）。[本次训练归档与导入校验](../TrainingRecords/Opus/opus-mixed-20261029-152352-250666/README.md)含每轮指标及备份位置。旧数据库及历史训练归档保留，原 HaiKu 续训记录使用 HaiKu 新库。

## 网络与接口

共享 CNN 与一次行列位置编码输出 `Z_t[B,90,128]`。五个核心参数独立，每个核心含两层历史编码、两层当前空间编码、一次内部多头交叉融合。

每个核心的五个 Block 均采用 `h = LN1(X + Dropout(GELU(AttentionOutput)))`、`output = LN2(h + Dropout(FFN(h)))`，FFN 为 `Linear(128,512) → GELU → Linear(512,128)`。每核心新增 10 个、全网 50 个 Dropout，默认 `p=0.1`，训练参数 `--dropout` 可覆盖并保存；旧配置缺省补为 0.1。Dropout 无参数，旧权重可严格加载，优化器参数顺序不变，eval 下与旧网络输出一致。监督训练开启，验证/对弈关闭；含有效 PPO 的逻辑批量关闭 Dropout 并保留梯度。

图示：[Opus 完整网络](diagrams/opus-5-dropout.svg)、[共享核心逐层结构](diagrams/shared-core-dropout-detailed.svg)；PNG 与全部三张图见 [图片目录](diagrams/README.md)。

| 核心 | 输入 | 含当前局面的 KV 窗口 | 内部头数 |
|---|---|---:|---:|
| 1 | Z_t | 10 | 16 |
| 2 | F1 | 20 | 8 |
| 3 | F2 | 30 | 4 |
| 4 | Z_t | 20 | 8 |
| 5 | P_t | 30 | 4 |

策略分支 `1 → 2 → 3`。价值分支先执行 4，再以 `Q=F4`、`K,V=F2` 做一次独立投影的单头交叉注意力，产生核心 5 的输入 P。之后以 `Q=F5`、`K,V=F3` 做另一次独立投影的单头交叉注意力，输出 O。两次跨分支注意力只读取当前时间步的 90 个位置，不增加时间缓存、残差或 FFN。

策略头从 F3 投影 Q、K，使用 `QKᵀ/√128` 的 `[B,90,90]` 原始评分，展平为 `[B,8100]` logits。屏蔽非法动作后，在全部 8100 个动作上统一 Softmax；不做 V 聚合，也不按起点逐行归一化。

价值头从 O 沿 90 个位置均值池化，再经过 `LayerNorm → Linear(128,64) → GELU → Linear(64,3)`，输出 logits。Softmax 后顺序为 **当前走方胜、和、负**，不是固定红方视角；标量价值为 `P胜-P负`。

```python
from model_store import load_default_model

net = load_default_model('Opus')  # 从该系列默认检查点恢复，不随机重建权重
out, cache = net.forward_step(x_t, cache=None)  # x_t: [B,15,10,9]
logits = out.policy_logits                     # [B,8100]
wdl = out.value_probabilities                  # [B,3]
value = out.expected_value                    # [B]
out = net(x_sequence, lengths)                # 策略 [T,B,8100]；价值 [T,B,3]
```

单局使用同一序列缓存；换局/换批次重建。缓存不写入模型权重。短局的填充步输出置零；训练损失也排除填充步。网络返回原始策略 logits，合法动作屏蔽在现有推理与损失入口执行。

## 训练与梯度截断

下面是正式 Opus 棋谱训练命令，CPU 环境将 `--extra gpu` 改为 `--extra cpu`，`--device cuda` 改为 `--device cpu`：

```powershell
uv run --locked --extra gpu xiangqi-train --architecture Opus --T 120 --context-windows 10 20 30 20 30 --tbptt-steps 16 --epochs 10 --batch-size 128 --micro-batch 1 --device cuda --description "Opus 首次棋谱策略训练"
```

`--heads 16 8 4 8 4` 可在新训练时显式指定；续训使用检查点头数。`--model-db` 省略时自动使用当前系列的独立库。续训使用 `--resume 实际参数ID`，评估使用 `--evaluate 实际参数ID`；可同时传 `--architecture Opus` 限定系列，未限定则从两个库查找。原 HaiKu 训练入口与旧导入路径仍兼容。

`T` 是古谱读取时的片段长度：完整棋局连续切段，尾段补零，每段有独立有效长度和真实起始棋盘。五个窗口 W 和梯度截断 K 与 T 独立。一个训练片段内部，每个反向传播段最多 K=16 步，计算损失并反向累积；段末 detach 全部 15 对 KV 缓存，保留数值与片段内时间。一个逻辑批量内所有微批量及所有反向传播段完成后才更新一次参数。窗口长于 16 时，后续反向传播段仍可读取当前片段内更早的缓存数值，但梯度不跨越反向传播段边界。不同训练片段独立建立缓存。

`forward_chunks` 是逐段生成器；自定义训练循环必须在取下一段前执行本段 backward，所有段完成后再 optimizer.step。直接 `net(sequence)` 提供完整前向接口，不隐式截断计算图；项目训练与评估入口已使用分段接口。

`python -m training --family Opus --source books` 读取古谱下一手与原始终局结果，默认同时优化策略交叉熵和当前走棋方胜/和/负交叉熵；未知结果只训练策略。按策略、价值、PPO、动作熵的顺序，古谱默认系数为 1/1/0/0，自弈和混合为 1/1/1/1。四项可通过对应 `--lambda-*` 独立覆盖。原 `train.py` 保留旧策略监督和优化器续训流程；Opus 策略与价值联合训练统一使用新入口。完整说明见 [混合训练与自弈接口](混合训练与自弈接口.md)。

桌面客户端自动显示当前局面的红方胜率、和棋概率、黑方胜率与简短说明。对弈模型有已训练的价值头时使用该模型；其他模式使用 Opus 后台默认 ID 对应的已训练价值模型，独立于走棋对手。模型输出始终以当前走棋方为视角，客户端按红黑换算。走子、悔棋、新局后清除旧预测并在后台重放历史重新评估。初始化或只训练过策略的旧参数不显示随机胜率；完成有价值监督的训练后，将所需 Opus ID 设为默认即可用于局势预测。

## 版本说明与默认加载

在项目根目录执行（有项目虚拟环境时使用 `.venv\Scripts\python.exe`，或加 `uv run --locked --extra cpu` 前缀）：

```powershell
python model_store.py list-versions
python model_store.py --family Opus list-versions
python model_store.py --family Opus describe --checkpoint 实际参数ID --version "Opus-5.1" --description "本次训练的数据与配置说明"
python model_store.py --family Opus set-default --checkpoint 实际参数ID
python model_store.py --family Opus publish --name "Opus-5" --checkpoint 实际参数ID
```

请替换“实际参数ID”。设置默认只改默认指针，修改版本说明只改元数据，都不覆盖任何权重。训练保存的新版本不会自动改变原默认指针或公开名称；`--set-default` 可让本次训练的 best 成为默认参数，但仍不自动发布。未训练或 `--smoke` 版本不可发布到游戏。

开发/训练代码调用 `load_default_model('HaiKu')` 或 `load_default_model('Opus')`，自动恢复各自默认参数。游戏默认汇总两个库的已发布名称，省略模型名称时优先选择可用的默认公开版本，图形界面仍需点击开始对弈。显式 `--model 名称` 使用对应名称的固定检查点。初始 Opus 供开发加载，不冒充已有棋力的正式对手。

复制服务器参数时分别复制相应系列的 SQLite 一致快照；不要用本地初始化库覆盖远端已训练库。原库迁移命令为 `python model_store.py migrate`，已有新 HaiKu 库时不会覆盖。新环境可用 `python model_store.py --family Opus initialize` 保存一份未训练初始化参数。
