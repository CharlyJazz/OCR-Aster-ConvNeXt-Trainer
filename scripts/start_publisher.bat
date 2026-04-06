@echo off
REM Run the HuggingFace -> Redis publisher
REM Usage: scripts\start_publisher.bat configs\training\my_config.yaml [--no-flush]

IF "%~1"=="" (
    echo Usage: start_publisher.bat ^<config_path^> [--no-flush]
    exit /b 1
)

cd /d %~dp0..
call .venv\Scripts\activate.bat
ocr-publish --config %*
