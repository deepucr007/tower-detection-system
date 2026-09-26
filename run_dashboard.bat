@echo off
cd /d "%~dp0"
title AI Tower Detection Dashboard
echo ========================================================
echo   AI TOWER DETECTION SYSTEM - DASHBOARD
echo ========================================================
echo.
echo Opening browser at http://127.0.0.1:8501 ...
echo.
start http://127.0.0.1:8501
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
pause
