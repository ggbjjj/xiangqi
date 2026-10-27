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
| Data/dataset.sqlite3 | 否 | **古谱训练必需，单独上传** | 数据未改变时不用取回 |
| ModelParameters/HaiKu/models.sqlite3 | 否 | 继续 HaiKu 训练时上传已有库 | **训练 HaiKu 后必需** |
| ModelParameters/Opus/models.sqlite3 | 否 | 继续 Opus 训练时上传已有库 | **训练 Opus 后必需** |
| Data/SelfPlay/HaiKu 或 Opus/games.sqlite3 | 否 | 需要使用已有自弈数据时上传 | 新生成或希望保留这些经验时取回 |
| runs/ 的本次运行目录 | 否 | 不要求旧日志；可选择上传查看历史 | **取回以便查看曲线、评优、配置和排查训练** |
| pretrain_data/ 原始下载 | 否，根 README 除外 | 只在重新导入棋谱时需要 | 通常不需要 |
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

在服务器终端执行：

```bash
cd /root/autodl-tmp
git clone --branch main https://github.com/ggbjjj/xiangqi.git xiangqi
cd xiangqi
uv sync --locked --python 3.12 --extra gpu
uv run --locked --extra gpu python tools/check_training_env.py
```

私有仓库需使用自己已配置的 GitHub 认证方式；也可使用本地相同的 SSH 443 仓库地址。Git clone 完成后还没有训练数据库和权重，这些文件需要单独上传到下列位置：

```text
/root/autodl-tmp/xiangqi/
├── Data/dataset.sqlite3
└── ModelParameters/
    ├── HaiKu/models.sqlite3
    └── Opus/models.sqlite3
```

只训练一个模型时只需该系列参数库。已有自弈经验用于训练时，另外上传 `Data/SelfPlay/对应模型/games.sqlite3`。不必上传本地虚拟环境；服务器用 GPU 依赖建立自己的环境。读取已有古谱库训练不需要 pretrain_data 原始下载。

使用 AutoDL 文件管理上传即可，也可从本地 PowerShell 使用 SCP（替换端口、用户名和服务器地址）：

```powershell
scp -P 实际SSH端口 'D:\象棋\Data\dataset.sqlite3' 用户名@服务器:/root/autodl-tmp/xiangqi/Data/
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

## 训练结束后给我哪些数据

需要两类文件：**对应模型的完整参数 SQLite + 本次完整运行日志**。参数 SQLite 已经包含所有 ID、累计完成轮数、权重、优化器、默认指针和评估指标；不需要再拆成 pt 文件。日志包含 config.json、metrics.jsonl、report.json、best.json（有完整验证评优时）、last_completed.json、tensorboard/ 等，整个本次运行目录取回即可。

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

exports 本身已被 Git 忽略。包内保留项目相对路径，解压到一个新目录后先核对内容；本地已有参数库时先备份，再选择要使用的那份库，不要直接覆盖或混合。不需要把未改变的 dataset.sqlite3、原始下载、虚拟环境或整个服务器项目再发回来。
