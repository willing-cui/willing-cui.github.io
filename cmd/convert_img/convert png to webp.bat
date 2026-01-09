@echo off
for %%a in (D:\GitHub\willing-cui.github.io\images\*.png) do (
    cwebp %%a -o D:\GitHub\willing-cui.github.io\images\%%~na.webp
    echo %%~na.webp
)