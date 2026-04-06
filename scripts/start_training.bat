@echo off
REM Run the training loop (reads from Redis)
REM Usage: scripts\start_training.bat configs\training\my_config.yaml

IF "%~1"=="" (
    echo Usage: start_training.bat ^<config_path^>
    exit /b 1
)

cd /d %~dp0..
call .venv\Scripts\activate.bat
ocr-train --config %*
