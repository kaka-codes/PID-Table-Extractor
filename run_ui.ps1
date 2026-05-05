$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path (Split-Path -Parent $repoRoot) "venv\Scripts\python.exe"
$appPath = Join-Path $repoRoot "app.py"

if (-not (Test-Path $pythonPath)) {
    throw "Expected Python environment was not found at '$pythonPath'."
}

if (-not (Test-Path $appPath)) {
    throw "Streamlit app was not found at '$appPath'."
}

& $pythonPath -m streamlit run $appPath @args
