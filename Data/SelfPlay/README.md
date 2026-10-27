# 自弈运行数据

HaiKu 与 Opus 分别使用 `HaiKu/games.sqlite3`、`Opus/games.sqlite3`，同系列不同参数版本共用该库。数据库及其临时侧文件不随 Git 提交。

新服务器不必携带空库，生成器会自动创建。继续使用既有自弈数据训练时需单独上传对应库；取回新生成的数据时可在导出命令添加 `--include-selfplay`。详见 [指令.md](../../指令.md)。
