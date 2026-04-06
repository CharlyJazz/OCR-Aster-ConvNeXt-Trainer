@echo off
REM Start publisher and trainer in separate terminals (same machine)
REM Usage: scripts\start_all.bat configs\training\my_config.yaml

IF "%~1"=="" (
    echo Usage: start_all.bat ^<config_path^>
    echo Example: start_all.bat configs\training\aster_v2_curriculum.yaml
    pause
    exit /b 1
)

SET CONFIG=%~1

cd /d %~dp0..

echo ========================================
echo  OCR-Aster Training  ^|  Redis Pipeline
echo ========================================
echo Config: %CONFIG%
echo.
echo Starting Publisher in a new terminal...
start "OCR Publisher" cmd /k "scripts\start_publisher.bat %CONFIG%"

echo Waiting 8 seconds for publisher to fill Redis...
timeout /t 8 /nobreak > nul

echo Starting Trainer in a new terminal...
start "OCR Trainer" cmd /k "scripts\start_training.bat %CONFIG%"

echo.
echo Both terminals opened.
echo Publisher feeds Redis ^| Trainer reads from Redis
echo Close each terminal with Ctrl+C to stop.
echo.
pause
