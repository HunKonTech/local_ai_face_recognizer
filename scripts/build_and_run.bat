@echo off
:: Build and run script for Windows

setlocal EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
set "VENV_DIR=%REPO_ROOT%\.venv"

pushd "%REPO_ROOT%"

:: ── Find Python 3.11+ ────────────────────────────────────────────────────────
echo =^> Checking Python...
set "PYTHON="

for %%c in (python3.13 python3.12 python3.11 python3 python) do (
    if not defined PYTHON (
        where %%c >nul 2>&1
        if not errorlevel 1 (
            for /f "tokens=*" %%v in ('%%c -c "import sys; ok = sys.version_info >= (3,11); print('ok' if ok else 'old')" 2^>nul') do (
                if "%%v"=="ok" set "PYTHON=%%c"
            )
        )
    )
)

if not defined PYTHON (
    echo ERROR: Python 3.11+ not found.
    echo        Install it from https://www.python.org/downloads/ and add it to PATH.
    exit /b 1
)

for /f "tokens=*" %%v in ('%PYTHON% -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%v
echo     Using Python %PY_VER% ^(%PYTHON%^)

:: ── Virtual environment ───────────────────────────────────────────────────────
echo =^> Setting up virtual environment at %VENV_DIR% ...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %PYTHON% -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

:: ── Dependencies ──────────────────────────────────────────────────────────────
echo =^> Installing / updating dependencies...
python -m pip install --upgrade pip setuptools wheel --quiet
python -m pip install -e ".[dev]" --quiet

echo =^> Build complete. Launching application...
python -m app.main %*

popd
endlocal
