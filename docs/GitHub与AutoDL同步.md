# GitHub、AutoDL 与训练结果同步

GitHub 仓库为 `ggbjjj/xiangqi`，当前本地分支为 main，origin 使用 SSH 443 地址。仓库负责同步代码、依赖版本和说明；训练数据库、参数、原始下载、日志和回传包通过 AutoDL 文件管理或 SCP 单独传输。

## 哪些进入 Git

| 内容 | Git 管理 | 首次发到 AutoDL | 训练后取回 |
|---|---|---|---|
| 根目录 Python、models/、training/、selfplay/、data_pipeline/、tools/、tests/ | 是 | git clone | 不必重复下载 |
| pyproject.toml、uv.lock、.python-version、requirements-training.txt | 是 | git clone | 不必重复下载 |
| 指令.md、README、docs/、现有课件图片与启动脚本 | 是 | git clone | 不必重复下载 |
| Data、ModelParameters 的 README 与小型元数据说明 | 是 | git clone | 按需 |
| TrainingRecords 的续训配置、摘要 JSON、说明 Markdown | 是 | git clone；旧入口使用这些记录 | 使用旧续训入口时保留更新后的记录 |
| Data/dataset.sqlite3 | 否 | **主棋谱库；--book-source main/both 时单独上传** | 数据未改变时不用取回 |
| Data/endgames.sqlite3 | 否 | **独立残局库；--book-source endgames/both 时单独上传** | 数据未改变时不用取回 |
| ModelParameters/HaiKu/models.sqlite3 | 否 | 继续 HaiKu 训练时上传已有库 | **训练 HaiKu 后必需** |
| ModelParameters/Opus/models.sqlite3 | 否 | 继续 Opus 训练时上传已有库 | **训练 Opus 后必需** |
| Data/SelfPlay/HaiKu 或 Opus/games.sqlite3 | 否 | 需要使用已有自弈数据时上传 | 新生成或希望保留这些经验时取回 |
| runs/ 的本次运行目录 | 否 | 不要求旧日志；可选择上传查看历史 | **取回以便查看曲线、评优、配置和排查训练** |
| pretrain_data/ 原始下载 | 否，根 README 除外 | 只在重新导入主棋谱或重建残局库时需要；使用现成 SQLite 训练不需要 | 通常不需要 |
| TrainingRecords/**/source/ 原始归档快照 | 否 | 通常不需要 | 需要完整原始归档时另行备份 |
| .venv、缓存、.idea/.vscode、build/dist、backups、artifacts、exports、文档/文档工作 | 否 | 不上传 | 不取回 |
| .env、SSH 私钥、SQLite 临时侧文件、权重二进制、下载压缩包 | 否 | 运行数据按对应流程传输 | 使用下面的数据库快照导出 |

`.gitignore` 控制这些规则；`.gitattributes` 使 Python、Shell、HTML、Markdown 等文本统一使用 LF，Windows 的 bat/ps1 使用 CRLF。已有课件图不变。IDE 文件若曾经被跟踪，需要取消跟踪，忽略规则才对后续修改生效；本地 IDE 文件保留。

## 本地提交与推送

在本地项目目录执行，先检查待提交清单：

```powershell
git status --short
git add -A
git diff --cached --stat
git commit -m 'Update model training and AutoDL workflow'
git push origin main
```

代码与元数据通过 Git 提交，数据库不会出现在提交中。修改忽略规则不会清除旧提交中已有的内容。本次配置不更换 origin、不重写已有历史。

## AutoDL 第一次拉取

选择服务器前先查看 [GPU 与 CUDA 兼容说明](../GPU与CUDA兼容说明.md)，核对显卡型号和驱动。当前 cu128 方案按 CUDA 12.8 Update 1 配套驱动基线部署：Linux 570.124.06、Windows 572.61 或更高，且驱动须支持具体显卡；CUDA 12.x 的兼容下限与使用限制在该文档中单独说明。

在服务器终端执行：

```bash
cd /root/autodl-tmp
git clone --branch main https://github.com/ggbjjj/xiangqi.git xiangqi
cd xiangqi
uv sync --locked --python 3.12 --extra gpu
uv run --locked --extra gpu python tools/check_training_env.py
```

普通依赖和 NVIDIA CUDA 运行库的阿里云镜像、2 路并发下载已写入 pyproject.toml，uv.lock 也已同步锁定实际下载地址。首次 clone 和后续 pull 后直接执行 `uv sync --locked --python 3.12 --extra gpu`，不需要每次设置镜像或运行 `uv lock`。GPU 版 PyTorch 本体单独使用南京大学 CUDA 12.8 镜像，CPU 版仍使用官方 CPU 索引。以上路径按服务器实际目录替换，例如当前服务器使用 `/hy-tmp/xiangqi`；旧环境首次升级的处理见下文。

私有仓库需使用自己已配置的 GitHub 认证方式；也可使用本地相同的 SSH 443 仓库地址。Git clone 完成后还没有训练数据库和权重，这些文件需要单独上传到下列位置：

```text
/root/autodl-tmp/xiangqi/
├── Data/
│   ├── dataset.sqlite3                # 主棋谱库，main/both 使用
│   ├── endgames.sqlite3               # 独立残局库，endgames/both 使用
│   └── SelfPlay/
│       ├── HaiKu/games.sqlite3        # 使用已有 HaiKu 自弈经验时上传
│       └── Opus/games.sqlite3         # 使用已有 Opus 自弈经验时上传
└── ModelParameters/
    ├── HaiKu/models.sqlite3           # 训练 HaiKu 时上传已有参数
    └── Opus/models.sqlite3            # 训练 Opus 时上传已有参数
```

根据训练命令的棋谱读取方式选择数据文件：

| 读取方式 | 训练参数 | 必须上传的数据文件 |
|---|---|---|
| 约 3 万局的主棋谱库 | `--book-source main`（默认） | `Data/dataset.sqlite3` |
| 仅独立残局库 | `--book-source endgames` | `Data/endgames.sqlite3` |
| 两库一起读取 | `--book-source both` | **`Data/dataset.sqlite3` 和 `Data/endgames.sqlite3` 两个都要上传** |

以上三种方式都通过同一读取函数加载，验证集跟随所选方式；完整训练命令见 [指令.md](../指令.md#三种棋谱加载命令)。`git pull` 不会下载或更新这两个被忽略的 SQLite。自定义位置时分别用 `--book-db` 和 `--endgame-db` 指定；旧 `train.py` 的主库路径参数为 `--data`。

只训练一个模型时只需该系列参数库。已有自弈经验用于训练时，另外上传 `Data/SelfPlay/对应模型/games.sqlite3`。纯自弈未启用棋谱验证时不需要这两个棋谱库。不必上传本地虚拟环境；服务器用 GPU 依赖建立自己的环境。读取已建好的主库或残局库训练都不需要 pretrain_data 原始下载，也不需要上传步数统计图和报告。

上传前，在服务器项目目录准备目标目录（项目路径按实际机器替换）：

```bash
cd /root/autodl-tmp/xiangqi
mkdir -p Data ModelParameters/HaiKu ModelParameters/Opus
```

使用 AutoDL 文件管理上传即可，也可从本地 PowerShell 使用 SCP（替换端口、用户名和服务器地址）：

```powershell
# 棋谱数据：main 上传第一份，endgames 上传第二份，both 两份都上传
scp -P 实际SSH端口 'D:\象棋\Data\dataset.sqlite3' 用户名@服务器:/root/autodl-tmp/xiangqi/Data/
scp -P 实际SSH端口 'D:\象棋\Data\endgames.sqlite3' 用户名@服务器:/root/autodl-tmp/xiangqi/Data/

# 初次部署时，只选择本次训练模型对应的参数库
scp -P 实际SSH端口 'D:\象棋\ModelParameters\HaiKu\models.sqlite3' 用户名@服务器:/root/autodl-tmp/xiangqi/ModelParameters/HaiKu/
scp -P 实际SSH端口 'D:\象棋\ModelParameters\Opus\models.sqlite3' 用户名@服务器:/root/autodl-tmp/xiangqi/ModelParameters/Opus/
```

已有服务器参数库时保留服务器版本，后续更新只拉代码；不要用较旧的本地库覆盖继续训练后的服务器库。两个地方从同一份库分别训练可能分配相同数字 ID，不能靠复制文件合并两个分支；将服务器作为这次训练的权威库，旧本地库另行备份。

上传后核对参数 ID，再按 [指令.md](../指令.md) 训练：

```bash
uv run --locked --extra gpu python model_store.py --family HaiKu list-parameters
uv run --locked --extra gpu python model_store.py --family Opus list-parameters
```

后续本地推送完成后，服务器更新代码：

```bash
cd /root/autodl-tmp/xiangqi
git pull --ff-only origin main
uv sync --locked --python 3.12 --extra gpu
```

在当前训练/生成任务结束后更新运行代码。被忽略的数据库与 runs 不会因普通 git pull 被覆盖；若服务器也改过源码，先处理这些改动再拉取。

## 已配置的依赖下载源

镜像配置随代码保存在 `pyproject.toml`，具体包版本、文件地址和校验值保存在 `uv.lock`，两份文件一起提交和拉取。

| 依赖 | 固定下载源 |
|---|---|
| NVIDIA CUDA 运行库：cuBLAS、cuDNN、NCCL、cuFFT、cuSPARSE 等全部 `nvidia-*` 依赖 | 阿里云 PyPI 镜像 `https://mirrors.aliyun.com/pypi/simple/` |
| NumPy、TensorBoard、Matplotlib、Triton 等其他普通依赖 | 同一阿里云 PyPI 镜像 |
| GPU 版 `torch==2.9.1+cu128` 本体 | 南京大学 PyTorch 镜像 `https://mirrors.nju.edu.cn/pytorch/whl/cu128` |
| 本地 CPU 版 `torch==2.9.1+cpu` 本体 | PyTorch 官方 CPU 索引 |

并发下载固定为 2。NVIDIA 依赖使用阿里云返回的 wheel 文件地址；镜像可能使用自己的 CDN，实际下载主机以 `uv.lock` 的 URL 为准。已经完整缓存的包可以复用，不需要删除 `.venv` 或清空缓存。

`pytorch-cu128` 设置为 `explicit = true`，仅供 `gpu` extra 的 `torch` 使用，不接管 NVIDIA 或其他依赖。锁文件同时保存南京大学镜像中的 Torch wheel 地址及与原官方文件一致的 SHA-256；PyTorch 版本和 CUDA 构建仍为 `2.9.1+cu128`。推送及拉取时必须包含 `pyproject.toml` 和 `uv.lock`，只改配置文件无法切换 `--locked` 使用的实际安装包地址。

### 旧服务器首次升级

曾在服务器修改过 `uv.lock`，且 `git pull` 提示该文件存在冲突时，先备份并检查差异：

```bash
cd /hy-tmp/xiangqi
mkdir -p backups/dependency-config
cp uv.lock "backups/dependency-config/uv.lock.$(date +%Y%m%d-%H%M%S)"
git diff -- uv.lock
```

确认差异只有此前更换下载源产生的修改后，执行 `git restore -- uv.lock`，再拉取；如果还修改了依赖版本或 `pyproject.toml`，先保留并处理这些差异。

旧终端中手动设置的索引环境变量会覆盖项目配置。首次升级时清除一次；如果曾写入 `~/.bashrc` 或 `~/.profile`，同时删除其中对应的设置行。以后新开终端直接使用项目配置：

```bash
unset UV_DEFAULT_INDEX UV_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_CONCURRENT_DOWNLOADS
git pull --ff-only origin main
uv sync --locked --python 3.12 --extra gpu
```

日常拉取代码后只需上述 `git pull` 和 `uv sync`；无需在服务器反复修改源或生成锁文件。启动中文曲线窗口也在项目根目录执行，确保 `runs` 指向训练日志目录：

```bash
uv run --locked --extra gpu python -m training.dashboard --logdir runs --host 127.0.0.1 --port 6006
```

参考：[uv 索引与优先级](https://docs.astral.sh/uv/concepts/indexes/)、[uv 锁文件与安装](https://docs.astral.sh/uv/concepts/projects/sync/)、[阿里云 PyPI 镜像](https://developer.aliyun.com/mirror/pypi)、[南京大学 PyTorch 镜像说明](https://help.mirror.nju.edu.cn/pytorch/)。

## 训练结束后给我哪些数据

需要两类文件：**对应模型的完整参数 SQLite + 本次完整运行日志**。参数 SQLite 已经包含所有 ID、累计完成轮数、权重、优化器、默认指针和评估指标；不需要再拆成 pt 文件。日志包含 config.json、metrics.jsonl、report.json、best.json（有完整验证评优时）、last_completed.json、tensorboard/ 及 **validation/**（逐片困惑度 JSONL、区间 PNG/SVG、每轮报告和 latest.json），整个本次运行目录取回。独立 `training.evaluate` 的输出另存于 `runs/evaluation/`，需要时一并取回。

| 目的 | 取回内容 |
|---|---|
| 加载/检查 HaiKu 新参数 | ModelParameters/HaiKu/models.sqlite3 |
| 加载/检查 Opus 新参数 | ModelParameters/Opus/models.sqlite3 |
| 看训练曲线、分析损失、核查最佳 ID 与中断进度 | 本次 runs/mixed/实际run_id/，或自定义的具体运行目录 |
| 分析新自弈棋局，或下次继续使用这些经验 | Data/SelfPlay/对应模型/games.sqlite3 |
| 使用旧 resume_train.py 继续原优化器训练 | 另保留相应 TrainingRecords 续训配置；新混合训练入口不依赖旧 current_resume.json |

先停止训练和生成，再导出。推荐用一条命令打包；数据库通过 SQLite backup API 生成独立快照，包含已经提交的 WAL 数据，不直接复制正在使用的主文件：

```bash
# HaiKu：参数 + 本次训练日志
uv run --locked --extra gpu python tools/export_training.py --family HaiKu --run-dir runs/mixed/实际run_id

# Opus：参数 + 本次训练日志 + 全部 Opus 自弈经验
uv run --locked --extra gpu python tools/export_training.py --family Opus --run-dir runs/mixed/实际run_id --include-selfplay
```

`--run-dir` 可多次指定以保留多次运行；原 train.py 日志也可以传具体运行目录。只取参数时省略 --run-dir。没有自弈库时省略 --include-selfplay。导出工具不依赖 PyTorch，不会修改源数据库，不会更新旧续训配置。

输出位置如下，时间按北京时间命名：

```text
exports/
├── return-Opus-时间戳/          # 可检查、可单独复制的快照目录
│   ├── ModelParameters/Opus/models.sqlite3
│   ├── Data/SelfPlay/Opus/games.sqlite3   # 添加选项时才有
│   ├── runs/mixed/实际run_id/
│   └── manifest.json          # 文件大小、SHA-256、代码提交号、是否有未提交修改
└── return-Opus-时间戳.tar.gz    # 下载这一个包即可交给我检查
```

exports 本身已被 Git 忽略。包内保留项目相对路径，解压到一个新目录后先核对内容；本地已有参数库时先备份，再选择要使用的那份库，不要直接覆盖或混合。导出工具默认不打包主棋谱库和残局库；训练只读它们，不需要把未改变的 `Data/dataset.sqlite3`、`Data/endgames.sqlite3`、原始下载、虚拟环境或整个服务器项目再发回来。若在服务器重新导入或修改了任一棋谱库，再单独取回该库。
