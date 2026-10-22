# 中国象棋

用 uv 管理的本地桌面游戏，支持双人、人机对弈（任选红黑）、AI 自动对弈。没有训练模型时使用内置搜索 AI。

## 启动

Windows 双击项目目录中的 `启动象棋.bat`。首次运行由 uv 安装 Python 3.12 和依赖，需要联网；以后复用 `.venv`。

也可以在项目目录执行：

```powershell
uv sync --locked
uv run xiangqi
```

主菜单数字键 1–4 选择模式。点击棋子，再点击绿色提示位置落子。
U 悔棋，N 新局，空格暂停/继续 AI 观战，B 或 Esc 返回菜单。
双人悔棋撤一步；人机悔棋撤销到玩家上一次落子前，AI 思考期间也可以悔棋或重新开始。
将死或困毙均判负。休闲规则：同局面、同走方第三次出现，或连续 120 步未吃子，判和；暂不包含正式比赛的长将、长捉裁决。

```powershell
uv run xiangqi gui --mode pve_red
uv run xiangqi gui --mode pve_black --depth 3
uv run xiangqi cli pvp
uv run xiangqi cli pve --ai-color b
```

命令行坐标为行、列，从 0 开始：`60-50`、`6,0 5,0`、`6050` 都表示红兵前进一步。输入 `u` 悔棋、`q` 退出。

## 神经网络与训练（可选）

正常游戏无需 PyTorch。已有模型或需要训练时安装 neural 扩展（默认 CPU 版）：

```powershell
uv sync --extra neural
uv run --extra neural xiangqi --help
uv run --extra neural xiangqi gui --model model.pt
uv run --extra neural xiangqi-train --iterations 1 --games-per-iter 1 --num-workers 1 --sims 10 --max-moves 20 --channels 16 --num-blocks 1 --epochs 1
```

`run_train.bat` / `run_train.ps1` 也可启动训练，支持传入上述参数。训练耗时取决于配置。
新模型自动保存网络结构信息；旧模型需用 `--channels`、`--blocks` 指定对应结构。

## 验证和构建

```powershell
uv run pytest -q
uv run --extra neural pytest -q
uv build
```

构建文件在 `dist/`，包含 wheel 和源码包；安装后使用 `xiangqi` 命令启动。
修改前源码备份在 `backups/original-project.zip`。
