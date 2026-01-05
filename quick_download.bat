@echo off
echo ====================================
echo FMA Medium Fast Download
echo ====================================
echo.
echo Installing aria2 (multi-connection downloader)...
winget install -e --id aria2.aria2

echo.
echo Starting download with 16 parallel connections...
aria2c -x 16 -s 16 -k 1M --file-allocation=none --dir=data\raw --out=fma_medium.zip https://os.unil.cloud.switch.ch/fma/fma_medium.zip

echo.
echo Download complete!
pause
