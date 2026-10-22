@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYGAME_HIDE_SUPPORT_PROMPT=1"
set "UV_EXE=uv"
where uv >nul 2>nul
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    ) else (
        echo 未找到 uv，请先安装 uv 后再运行。
        echo 安装说明：https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
)
echo 正在启动中国象棋，首次运行需要安装环境，请稍候...
if exist "model.pt" (
    "%UV_EXE%" run --locked --no-dev --extra neural xiangqi %*
) else (
    "%UV_EXE%" run --locked --no-dev xiangqi %*
)
if errorlevel 1 (
    echo.
    echo 启动失败，请查看上方错误；首次运行请确认网络可用。
    pause
    exit /b 1
)
