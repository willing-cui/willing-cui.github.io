@echo off
for %%a in (D:\GitHub\willing-cui.github.io\images\earth\*.png) do (
    cwebp %%a -o D:\GitHub\willing-cui.github.io\images\earth\%%~na.webp
    echo %%~na.webp
)