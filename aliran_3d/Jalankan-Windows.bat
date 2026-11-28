@echo off
setlocal
cd /d "%~dp0"
title Aliran 3D Bentheimer

rem 1) Python launcher (py) bila terpasang
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)" >nul 2>nul
if not errorlevel 1 (
    py -3 server.py
    goto selesai
)

rem 2) python biasa (alias Microsoft Store yang kosong akan gagal di uji ini)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)" >nul 2>nul
if not errorlevel 1 (
    python server.py
    goto selesai
)

rem 3) Tanpa Python: server bawaan PowerShell
echo Python tidak ditemukan, memakai server PowerShell.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0server.ps1"

:selesai
if errorlevel 1 pause
endlocal
