$env:PYTHONUTF8 = "1"
Push-Location $PSScriptRoot
try {
    uv run --locked --extra cpu xiangqi-train @args 2>&1 | Tee-Object -FilePath "train.log"
    $trainExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $trainExitCode
