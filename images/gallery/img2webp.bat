@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

title Image to WebP Converter
set "CURRENT_DIR=%~dp0"

:: Ask for folder name
echo ==========================================
echo    Image to WebP Converter
echo ==========================================
echo Supported formats: PNG, JPG, JPEG, JPE, JFIF
echo.

:ASK_FOLDER
set "FOLDER_NAME="
set /p "FOLDER_NAME=Please enter the folder name: "

if "%FOLDER_NAME%"=="" (
    echo Error: Folder name cannot be empty!
    goto ASK_FOLDER
)

if "%FOLDER_NAME:~-1%"=="\" set "FOLDER_NAME=%FOLDER_NAME:~0,-1%"
set "TARGET_PATH=%CURRENT_DIR%%FOLDER_NAME%"

if not exist "%TARGET_PATH%\" (
    echo Error: Folder not found: %TARGET_PATH%
    goto ASK_FOLDER
)

echo.
set "QUALITY=85"
set /p "QUALITY=Enter quality (1-100, default 85): "
if "%QUALITY%"=="" set "QUALITY=85"

echo.
set "DELETE_FLAG=0"
set /p "DELETE_OPT=Delete original files? (y/n, default n): "
if /i "%DELETE_OPT%"=="y" set "DELETE_FLAG=1"

echo.
set "RECURSIVE_FLAG="
set /p "RECURSIVE=Process subfolders? (y/n, default n): "
if /i "%RECURSIVE%"=="y" set "RECURSIVE_FLAG=/s"

echo.
echo ==========================================
echo Settings:
echo Source: %TARGET_PATH%
echo Quality: %QUALITY%
echo Delete original: %DELETE_OPT%
echo Recursive: %RECURSIVE%
echo ==========================================
echo.

where cwebp >nul 2>nul
if errorlevel 1 (
    echo Error: cwebp.exe not found in PATH!
    echo Please install WebP tools or add to PATH.
    pause
    exit /b 1
)

echo Searching for image files...
echo.

set "TOTAL=0"
set "PNG_COUNT=0"
set "JPG_COUNT=0"
set "SUCCESS=0"
set "FAILED=0"

:: 切换到目标目录进行处理
pushd "%TARGET_PATH%"

:: Process PNG files
for %%e in (png) do (
    for /f "delims=" %%f in ('dir /b %RECURSIVE_FLAG% "*.%%e" 2^>nul') do (
        set /a "TOTAL+=1"
        set /a "PNG_COUNT+=1"
        
        set "INPUT_FILE=%%f"
        set "OUTPUT_FILE=%%~nf.webp"
        
        echo [!TOTAL!] Processing: %%f
        
        if exist "!OUTPUT_FILE!" (
            echo    Skipped: WebP file already exists
            set /a "FAILED+=1"
        ) else (
            cwebp "!INPUT_FILE!" -q %QUALITY% -o "!OUTPUT_FILE!"
            
            if !errorlevel! equ 0 (
                echo    Success: !OUTPUT_FILE!
                set /a "SUCCESS+=1"
                
                if !DELETE_FLAG! equ 1 (
                    del "!INPUT_FILE!"
                    echo    Deleted: %%f
                )
            ) else (
                echo    Failed: Conversion error
                set /a "FAILED+=1"
            )
        )
        echo.
    )
)

:: Process JPG files
for %%e in (jpg jpeg jpe jfif) do (
    for /f "delims=" %%f in ('dir /b %RECURSIVE_FLAG% "*.%%e" 2^>nul') do (
        set /a "TOTAL+=1"
        set /a "JPG_COUNT+=1"
        
        set "INPUT_FILE=%%f"
        set "OUTPUT_FILE=%%~nf.webp"
        
        echo [!TOTAL!] Processing: %%f
        
        if exist "!OUTPUT_FILE!" (
            echo    Skipped: WebP file already exists
            set /a "FAILED+=1"
        ) else (
            cwebp "!INPUT_FILE!" -q %QUALITY% -o "!OUTPUT_FILE!"
            
            if !errorlevel! equ 0 (
                echo    Success: !OUTPUT_FILE!
                set /a "SUCCESS+=1"
                
                if !DELETE_FLAG! equ 1 (
                    del "!INPUT_FILE!"
                    echo    Deleted: %%f
                )
            ) else (
                echo    Failed: Conversion error
                set /a "FAILED+=1"
            )
        )
        echo.
    )
)

:: 切回原始目录
popd

echo ==========================================
echo Conversion Complete!
echo ==========================================
echo Statistics:
echo PNG files: %PNG_COUNT%
echo JPG files: %JPG_COUNT%
echo Total files: %TOTAL%
echo Successful: %SUCCESS%
echo Failed: %FAILED%
echo Quality: %QUALITY%
echo ==========================================
echo.

if %TOTAL% equ 0 (
    echo No image files found in: %TARGET_PATH%
    echo Supported formats: PNG, JPG, JPEG, JPE, JFIF
    echo.
)

echo Output folder: %TARGET_PATH%
pause