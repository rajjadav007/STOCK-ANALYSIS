@echo off
echo ================================================
echo Starting Stock Analysis Dashboard
echo ================================================
echo.

echo Starting Flask Backend API on port 5000...
start "Flask API" cmd /k "python dashboard_api.py"

timeout /t 3 /nobreak > nul

echo Starting React Dashboard on port 3000...
cd dashboard
start "React Dashboard" cmd /k "npm start"

echo.
echo ================================================
echo Both services are starting...
echo.
echo Backend API: http://localhost:5000
echo Dashboard: http://localhost:3000
echo.
echo Press any key to exit (this will NOT stop the services)
echo ================================================
pause > nul
