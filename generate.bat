@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ============================================
echo   CourtVisionNet - Synthetic Data Generator
echo ============================================
echo.
echo  [1] Render images with Blender
echo  [2] Convert raw output to CVN format
echo  [3] Render then convert
echo.
set /p "CHOICE=Choose (1/2/3): "

if "%CHOICE%"=="1" goto render_settings
if "%CHOICE%"=="2" goto convert_settings
if "%CHOICE%"=="3" goto render_settings_then_convert
echo Invalid choice.
goto done

:render_settings
set "DO_CONVERT=0"
goto ask_render

:render_settings_then_convert
set "DO_CONVERT=1"
goto ask_render

:ask_render
echo.
echo --- Render Settings (press Enter for default) ---
echo.

set "R_COUNT=500"
set /p "R_COUNT=  Image count [500]: "

set "R_SEED=42"
set /p "R_SEED=  Random seed (-1 for random) [42]: "

set "R_ENGINE=BLENDER_EEVEE"
set /p "R_ENGINE=  Engine (BLENDER_EEVEE / CYCLES) [BLENDER_EEVEE]: "

set "R_SAMPLES=32"
set /p "R_SAMPLES=  Render samples [32]: "

set "R_RESMIN=800"
set /p "R_RESMIN=  Min resolution [800]: "

set "R_RESMAX=1280"
set /p "R_RESMAX=  Max resolution [1280]: "

set "R_START=1"
set /p "R_START=  Start index [1]: "

echo.
echo --- Settings ---
echo   Count:      !R_COUNT!
echo   Seed:       !R_SEED!
echo   Engine:     !R_ENGINE!
echo   Samples:    !R_SAMPLES!
echo   Resolution: !R_RESMIN! - !R_RESMAX!
echo   Start:      !R_START!
echo.

python scripts/generate.py render --count !R_COUNT! --seed !R_SEED! --engine !R_ENGINE! --samples !R_SAMPLES! --res-min !R_RESMIN! --res-max !R_RESMAX! --start !R_START!

if errorlevel 1 (
    echo.
    echo Render failed with error code %errorlevel%.
    if "%DO_CONVERT%"=="1" echo Skipping conversion.
    goto done
)

if "%DO_CONVERT%"=="1" (
    echo.
    echo === Render complete, starting conversion ===
    echo.
    python scripts/generate.py convert
)

goto done

:convert_settings
echo.
echo --- Convert Settings (press Enter for default) ---
echo.

set "C_MINVIS=4"
set /p "C_MINVIS=  Min visible keypoints [4]: "

echo.
python scripts/generate.py convert --min-visible !C_MINVIS!
goto done

:done
echo.
echo ============================================
pause
