@echo off
echo ========================================
echo   Stock Analysis ML Trading System
echo ========================================
echo.
echo Starting Backend API (Flask)...
start "Backend API" cmd /k "cd backend && D:/STOCK-ANALYSIS/.venv/Scripts/python.exe dashboard_api.py"
echo.
timeout /t 3 /nobreak > nul
echo Starting Frontend Dashboard (React)...
start "Frontend Dashboard" cmd /k "cd frontend && npm start"
echo.
echo ========================================
echo Services Starting...
echo.
echo Backend API: http://localhost:5000
echo Frontend Dashboard: http://localhost:3000
echo.
echo Wait 5-10 seconds for services to start
echo ========================================
pause
