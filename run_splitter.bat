@echo off
REM -----------------------------------------------------------------------
REM  Double-click to run the model splitter interactively.
REM  Or drag an STL/3MF onto this .bat to pre-fill the input file.
REM -----------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv not found. Run this once from a terminal:
    echo     python -m venv venv
    echo     venv\Scripts\pip install numpy trimesh manifold3d shapely pillow rtree networkx lxml scipy
    echo.
    pause
    exit /b 1
)

"venv\Scripts\python.exe" split_model.py %*

echo.
echo --------------------------------------------------
echo Finished. Press any key to close this window.
pause >nul
